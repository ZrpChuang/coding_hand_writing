"""Scaled Dot-Product Attention：带逐行 Python 语法和逻辑注释的手撕版。"""

# math 是 Python 标准数学库，这里使用 sqrt 计算缩放因子。
import math
# torch 提供 Tensor、矩阵乘法和 softmax 等运算。
import torch


# def 用来定义函数；mask=None 表示 mask 是可选参数，默认值为空。
def attention(q, k, v, mask=None):
    """q:[B,Lq,Dk]，k:[B,Lk,Dk]，v:[B,Lk,Dv]；mask 中 0 表示不可见。"""
    # size(-1) 取得 Q 最后一维 Dk；transpose 交换 K 的最后两维。
    scores = q @ k.transpose(-2, -1)
    # 除以 sqrt(Dk)，防止点积随着维度增大而过大。
    scores = scores / math.sqrt(q.size(-1))

    # is not None 判断调用者有没有传入 mask。
    if mask is not None:
        # == 生成布尔张量；masked_fill 在 True 位置填入负无穷。
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # dim=-1 表示沿最后一维归一化，即每个 query 对所有 key 的权重和为 1。
    weights = torch.softmax(scores, dim=-1)
    # @ 是矩阵乘法：[B,Lq,Lk] @ [B,Lk,Dv] -> [B,Lq,Dv]。
    output = weights @ v
    # return 返回一个元组，包含注意力输出和注意力权重。
    return output, weights


# 只有直接执行本文件时才运行 demo；被其他文件 import 时不会运行。
if __name__ == "__main__":
    # 创建 Q；2 是 batch，3 是 query 数量，4 是 Q/K 的特征维度。
    q = torch.randn(2, 3, 4)
    # 创建 K；key 数量可以和 query 数量不同，但特征维度必须同为 4。
    k = torch.randn(2, 5, 4)
    # 创建 V；V 的 token 数必须和 K 相同，特征维度 Dv 可以不同。
    v = torch.randn(2, 5, 6)
    # 创建全 1 mask；其形状与注意力分数 [B,Lq,Lk] 相同。
    mask = torch.ones(2, 3, 5)
    # 冒号表示选中该维全部元素；-1 表示最后一个 key，将它设为不可见。
    mask[:, :, -1] = 0

    # 调用 attention；利用元组解包接收两个返回值。
    output, weights = attention(q, k, v, mask)
    # 输出形状由 query 数 Lq 和 value 维度 Dv 决定。
    print("output shape:", output.shape)       # [2,3,6]
    # 权重形状表示每个 query 对每个 key 的注意力大小。
    print("weights shape:", weights.shape)     # [2,3,5]

