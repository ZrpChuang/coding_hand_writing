"""Qwen3-VL Interleaved M-RoPE 手撕版：T/H/W 在频率维交错。"""

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def interleaved_mrope(q, k, position_ids_3d, sections, base=10000.0):
    """q/k: [B,H,L,D]；position_ids_3d: [3,B,L]，依次为 T/H/W。"""
    dim = q.size(-1)
    assert sum(sections) == dim // 2
    inv_freq = 1.0 / base ** (
        torch.arange(0, dim, 2, device=q.device).float() / dim
    )

    freqs_3d = position_ids_3d[..., None].float() * inv_freq

    # 默认全部取 T，再在下标 1,4,7... 和 2,5,8... 填入 H、W。
    # 得到布局：THWTHW...TT，同时保持频率从高到低连续。
    freqs = freqs_3d[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        end = sections[axis] * 3
        freqs[..., offset:end:3] = freqs_3d[axis, ..., offset:end:3]

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

    q, k = interleaved_mrope(q, k, position_ids_3d, sections=(2, 2, 2))
    print(q.shape, k.shape)                   # [1,2,6,12]

