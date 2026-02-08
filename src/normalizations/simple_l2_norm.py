import torch
import torch.nn as nn

class SimpleL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps  # Epsilon value to avoid division by zero during normalization

    def forward(self, x):
        # Normalize along the last dimension (head_dim)
        # This function normalizes the input tensor 'x' along its last dimension.
        # It computes the L2 norm (Euclidean norm) of 'x' and scales 'x' by the inverse of this norm.
        # The epsilon value is added to the denominator to ensure numerical stability and avoid division by zero.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
