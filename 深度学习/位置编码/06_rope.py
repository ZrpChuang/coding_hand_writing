"""传统 RoPE 手撕版。"""

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope(q, k, position_ids, base=10000.0):
    """q/k: [B,H,L,D]；position_ids: [B,L]。"""
    dim = q.size(-1)
    inv_freq = 1.0 / base ** (
        torch.arange(0, dim, 2, device=q.device).float() / dim
    )
    freqs = position_ids[..., None].float() * inv_freq       # [B,L,D/2]
    angles = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)  # [B,1,L,D]
    cos, sin = angles.cos(), angles.sin()

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


if __name__ == "__main__":
    q = torch.randn(1, 2, 6, 8)              # [B,H,L,D]
    k = torch.randn(1, 2, 6, 8)
    position_ids = torch.arange(6)[None, :]   # [B,L]

    q, k = rope(q, k, position_ids)
    print(q.shape, k.shape)                   # [1,2,6,8]

