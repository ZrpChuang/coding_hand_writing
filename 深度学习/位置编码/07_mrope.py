"""Qwen2-VL M-RoPE 手撕版：频率维按 T/H/W 连续分块。"""

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def mrope(q, k, position_ids_3d, sections, base=10000.0):
    """q/k: [B,H,L,D]；position_ids_3d: [3,B,L]，依次为 T/H/W。"""
    dim = q.size(-1)
    assert sum(sections) == dim // 2
    inv_freq = 1.0 / base ** (
        torch.arange(0, dim, 2, device=q.device).float() / dim
    )

    # 三套位置分别乘相同频率：[3,B,L,D/2]
    freqs_3d = position_ids_3d[..., None].float() * inv_freq
    t_freqs, h_freqs, w_freqs = freqs_3d
    t, h, _ = sections

    # 连续分块：TT...HH...WW...
    freqs = torch.cat((
        t_freqs[..., :t],
        h_freqs[..., t:t + h],
        w_freqs[..., t + h:],
    ), dim=-1)

    angles = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    cos, sin = angles.cos(), angles.sin()
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


if __name__ == "__main__":
    q = torch.randn(1, 2, 6, 12)
    k = torch.randn(1, 2, 6, 12)
    position_ids_3d = torch.tensor([
        [[0, 0, 0, 0, 1, 1]],               # T
        [[0, 0, 1, 1, 0, 0]],               # H
        [[0, 1, 0, 1, 0, 1]],               # W
    ])

    q, k = mrope(q, k, position_ids_3d, sections=(2, 2, 2))
    print(q.shape, k.shape)                   # [1,2,6,12]

