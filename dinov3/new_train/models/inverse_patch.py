import torch
import torch.nn as nn


class InversePatchEmbeddingMLP(nn.Module):
    """
    Input:  x [B, 196, D]
    Output: g [B, D]
    """
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim // 2

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=14, padding=0),
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        conv_layers = [m for m in self.net if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            return

        # Conservative init for early layers and zero-init last projection for stability.
        for module in conv_layers[:-1]:
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        last = conv_layers[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

        if isinstance(self.norm, nn.LayerNorm):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        B, N, D = x.shape
        assert N == 196, "N must be 196 (14x14)"

        out_dtype = x.dtype
        conv_weight = self.net[0].weight
        x = torch.nan_to_num(
            x.to(dtype=conv_weight.dtype, device=conv_weight.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # [B, 196, D] -> [B, D, 14, 14]
        x = x.view(B, 14, 14, D).permute(0, 3, 1, 2)

        # [B, D, 1, 1]
        g = self.net(x)
        

        # [B, D]
        g = g.squeeze(-1).squeeze(-1)

        g_norm = self.norm(g)
        g_norm = torch.nan_to_num(g_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return g_norm.to(dtype=out_dtype)
