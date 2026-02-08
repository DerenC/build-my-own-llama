import torch

from simplified_llama4_attention import SimplifiedLlama4Attention
from simplified_llama4_ffn import SimplifiedLlama4FFN

## ATTENTION MODULE

batch_size = 2
seq_len = 11
token_vec_emb_size = 128

token_vec_embs = torch.randn(batch_size, seq_len, token_vec_emb_size)

attn_configs_dict = {
    "token_vec_emb_size": token_vec_emb_size,
    "q_dim": 128,
    "num_query_heads": 16,
    "num_key_value_heads": 4,
    "max_seq_len": 256, # This can just be the sequence length, if known beforehand.
    "rope_theta": 10000.0,
    "using_attn_bias": False,
    "attn_dropout": 0.0,
    "normalising_qk": True,
    "training": False,  # False means inference
}

attn_module = SimplifiedLlama4Attention(attn_configs_dict)

# Run forward pass
attn_output, attn_weights = attn_module(token_vec_embs)

print("\nOutput shape from simplified module:", attn_output.shape)
print("Attention weights shape from simplified module:", attn_weights.shape)

## FFN MODULE

ffn_input_block = attn_output

hidden_size = token_vec_emb_size

ffn_intermediate_ratio = 8 / 3
intm_size_multiplier = 32
intermediate_size = int(hidden_size * ffn_intermediate_ratio)
intermediate_size = ((intermediate_size + intm_size_multiplier - 1) // intm_size_multiplier) * intm_size_multiplier

ffn_configs_dict = {
    'hidden_size': hidden_size,
    'intermediate_size': intermediate_size,
    'hidden_act_name': "silu",
    'using_ffn_bias': False,
    'rms_norm_eps': 1e-5,
}

ffn_module = SimplifiedLlama4FFN(ffn_configs_dict)
ffn_output = ffn_module(ffn_input_block)

# Residual Connection
# The output of FFN is added to the input of FFN
final_output = ffn_input_block + ffn_output

print("\nOutput shape from simplified FFN module (before residual):", ffn_output.shape)
print("Output shape after external residual connection:", final_output.shape)

# Verify that the manual calculation matches the output of the two modules (should be very close)
print("Outputs are close:", torch.allclose(attn_output, final_output, atol=1e-6))
