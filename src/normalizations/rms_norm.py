import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # Learnable gain parameter
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32) # Calculate in float32 for stability
        # Calculate variance (mean of squares) across the hidden dimension
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Normaliza: input / sqrt(variance + epsilon)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Apply learnable weight and cast back to original dtype
        return (self.weight * hidden_states).to(original_dtype)
