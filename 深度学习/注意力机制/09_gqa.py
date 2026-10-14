"""Grouped-Query Attention：多个 Q 头共享较少的 K/V 头。"""

# math 提供平方根函数。
import math
# torch 提供张量和矩阵运算。
import torch
# nn 提供 Linear、Module 等神经网络组件。
from torch import nn


# 定义 GQA 网络层；它是 nn.Module 的子类。
class GroupedQueryAttention(nn.Module):
    # num_q_heads 是 Q 头数，num_kv_heads 是共享的 K/V 头数。
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        # 初始化父类 nn.Module。
        super().__init__()
        # Q 的总维度必须可以平均分给所有 Q 头。
        assert d_model % num_q_heads == 0
        # Q 头数必须是 K/V 头数的整数倍，才能均匀共享。
        assert num_q_heads % num_kv_heads == 0

        # 保存 Q 头数量。
        self.num_q_heads = num_q_heads
        # 保存 K/V 头数量。
        self.num_kv_heads = num_kv_heads
        # 每一个注意力头的维度。
        self.head_dim = d_model // num_q_heads
        # 每个 K/V 头需要复制的次数。
        self.num_groups = num_q_heads // num_kv_heads

        # Q 仍然投影成 num_q_heads 个头，总输出维度等于 d_model。
        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim)
        # K 只投影成较少的 num_kv_heads 个头，从而减少参数和 KV Cache。
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        # V 与 K 使用相同数量的头。
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        # 最终把所有 Q 头的结果投影回 d_model。
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, d_model)

    # 把最后一维拆成多个注意力头。
    def split_heads(self, x, num_heads):
        # 读取 batch、序列长度和投影维度。
        batch_size, seq_len, _ = x.shape
        # view：[B,L,H*Dh] -> [B,L,H,Dh]。
        x = x.view(batch_size, seq_len, num_heads, self.head_dim)
        # transpose：[B,L,H,Dh] -> [B,H,L,Dh]。
        return x.transpose(1, 2)

    # forward 定义 GQA 的前向计算。
    def forward(self, x, mask=None):
        # Q 拆成 num_q_heads 个头。
        q = self.split_heads(self.q_proj(x), self.num_q_heads)
        # K 只拆成 num_kv_heads 个头。
        k = self.split_heads(self.k_proj(x), self.num_kv_heads)
        # V 只拆成 num_kv_heads 个头。
        v = self.split_heads(self.v_proj(x), self.num_kv_heads)

        # repeat_interleave 沿 head 维复制，让每组 Q 头共享同一个 K 头。
        k = k.repeat_interleave(self.num_groups, dim=1)
        # V 使用相同方式复制到 num_q_heads 个头。
        v = v.repeat_interleave(self.num_groups, dim=1)

        # QK^T 得到注意力分数，并除以 sqrt(Dh) 做缩放。
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # 如果提供了 mask，就屏蔽不可见位置。
        if mask is not None:
            # mask 为 0 的分数填负无穷，softmax 后权重变为 0。
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # 对每个 query 的所有 key 分数做归一化。
        weights = torch.softmax(scores, dim=-1)
        # 权重乘 V 得到每个 Q 头的输出。
        output = weights @ v

        # 读取 batch 和序列长度；下划线表示该维数值不需要使用。
        batch_size, _, seq_len, _ = output.shape
        # 先换回 [B,L,H,Dh]，contiguous 后再合并最后两维。
        output = output.transpose(1, 2).contiguous()
        # view 把多个头拼成 [B,L,H*Dh]。
        output = output.view(batch_size, seq_len, -1)
        # 输出投影融合所有 Q 头。
        output = self.out_proj(output)
        # 返回最终输出和注意力权重。
        return output, weights


# 直接运行本文件时执行 demo。
if __name__ == "__main__":
    # B=2、L=4、D=8 的随机输入。
    x = torch.randn(2, 4, 8)
    # 创建 causal mask，保证当前位置只能看到自己和之前的位置。
    mask = torch.tril(torch.ones(4, 4))[None, None, :, :]
    # 4 个 Q 头共享 2 个 K/V 头，每个 K/V 头服务 2 个 Q 头。
    gqa = GroupedQueryAttention(d_model=8, num_q_heads=4, num_kv_heads=2)
    # 执行前向计算，并用元组解包接收结果。
    output, weights = gqa(x, mask)
    # 最终输出形状仍然与输入相同。
    print("output shape:", output.shape)       # [2,4,8]
    # 注意力权重拥有 4 个 Q 头。
    print("weights shape:", weights.shape)     # [2,4,4,4]

