import torch
import torch.nn as nn
import math

from rope_funcs import get_rope_freqs_cis, apply_rope_to_qk
from simple_l2_norm import SimpleL2Norm

def repeat_kv_gpa(states_matrix: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1: return states_matrix

    batch, num_key_value_heads, seq_len, head_dim = states_matrix.shape
    states_matrix = states_matrix[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return states_matrix.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

required_configs = ("token_vec_emb_size", "q_dim", "num_query_heads", "max_seq_len")

class SimplifiedLlama4Attention(nn.Module):
    def __init__(self, configs):
        super().__init__()
        assert all(config in configs for config in required_configs)
        self.token_vec_emb_size = configs['token_vec_emb_size']
        self.q_dim = configs['q_dim']
        self.num_query_heads = configs['num_query_heads']

        self.head_dim = self.token_vec_emb_size // self.num_query_heads
        if self.head_dim % 2:
            raise ValueError("The attention head size should be an even number")
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_key_value_heads = configs.get('num_key_value_heads', self.num_query_heads)
        self.num_key_value_groups = self.num_query_heads // self.num_key_value_heads

        self.max_seq_len = configs['max_seq_len']
        self.rope_theta = configs.get('rope_theta', 10000.0)
        self.using_attn_bias = configs.get('using_attn_bias', False)
        self.attn_dropout = configs.get('attn_dropout', 0.0)
        self.normalising_qk = configs.get('normalising_qk', True)
        self.training = configs.get('training', False)  # Inference by default

        self.query_proj = nn.Linear(self.token_vec_emb_size, self.num_attention_heads * self.head_dim, bias=self.using_attn_bias)
        self.key_proj = nn.Linear(self.token_vec_emb_size, self.num_key_value_heads * self.head_dim, bias=self.using_attn_bias)
        self.value_proj = nn.Linear(self.token_vec_emb_size, self.num_key_value_heads * self.head_dim, bias=self.using_attn_bias)
        self.output_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.token_vec_emb_size, bias=self.using_attn_bias)

        self.freqs_cis = get_rope_freqs_cis(self.head_dim, self.max_seq_len, base=self.rope_theta)

        if self.normalising_qk: self.qk_norm = SimpleL2Norm()

    def forward(self, token_vec_embs):
        batch_size, seq_len, token_emb_size = token_vec_embs.shape
        if token_emb_size != self.token_vec_emb_size:
            raise ValueError("The input token embeddings do not have the same size as that in the implementation of this attention module")
        if seq_len > self.max_seq_len:
            raise ValueError("The input sequence length exceeds the maximal sequence length")

        position_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * -torch.inf, diagonal=1)

        # Projections
        query_states = self.query_proj(token_vec_embs)
        key_states = self.key_proj(token_vec_embs)
        value_states = self.value_proj(token_vec_embs)

        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        current_freqs_cis = self.freqs_cis.to(token_vec_embs.device) # Get precomputed freqs
        current_freqs_cis = current_freqs_cis[position_ids]
        current_freqs_cis = current_freqs_cis[:, None, :, :]
        query_states_rope, key_states_rope = apply_rope_to_qk(query_states, key_states, current_freqs_cis)

        # Optional QK Norm
        if self.use_qk_norm:
            query_states_final = self.qk_norm(query_states_rope)
            key_states_final = self.qk_norm(key_states_rope)
        else:
            query_states_final = query_states_rope
            key_states_final = key_states_rope

        # Repeat K/V for GQA
        key_states_repeated = repeat_kv_gpa(key_states_final, self.num_key_value_groups)
        value_states_repeated = repeat_kv_gpa(value_states, self.num_key_value_groups)

        # Attention Calculation
        attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))
        scaling_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (attn_weights * scaling_factor) + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

        # Dropout (during training)
        if self.attn_dropout > 0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states_repeated)

        # Reshape and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, self.hidden_size)
        final_attn_output = self.output_proj(attn_output)

        return final_attn_output, attn_weights # Return weights for inspection

config_dict = {
    "token_vec_emb_size": 128,
    "q_dim": 128,
    "num_query_heads": 16,
    "num_key_value_heads": 4,
    "max_seq_len": 256, # This can just be the sequence length, if known beforehand.
    "rope_theta": 10000.0,
    "using_attn_bias": False,
    "attn_dropout": 0.0,
    "normalising_qk": True,
    "training": False,
}