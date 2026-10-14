"""Multi-Head Self-Attention：带逐行 Python 语法和逻辑注释的手撕版。"""

# math 是 Python 标准数学库，这里使用 sqrt 计算缩放因子。
import math
# torch 提供 Tensor、矩阵乘法、softmax 等深度学习运算。
import torch
# from ... import ... 表示从 torch 中只导入 nn 模块。
from torch import nn


# class 用来定义类；继承 nn.Module 后，该类才能作为 PyTorch 网络层使用。
class MultiHeadAttention(nn.Module):
    # __init__ 是构造函数；self 表示当前对象；d_model 是总维度，num_heads 是头数。
    def __init__(self, d_model, num_heads):
        # super() 取得父类对象，调用父类构造函数以初始化 nn.Module。
        super().__init__()
        # assert 表示条件必须成立；总维度必须能被头数整除。
        assert d_model % num_heads == 0

        # self.xxx 把变量保存为对象属性，之后其他成员函数也可以访问。
        self.num_heads = num_heads
        # // 是整除；每个注意力头的维度 Dh = D / H。
        self.head_dim = d_model // num_heads
        # 保存总维度，合并多个头时会用到。
        self.d_model = d_model

        # nn.Linear(D,D) 创建线性层，分别把输入映射成 Q、K、V。
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 多个头拼接之后，再通过输出投影融合各个头的信息。
        self.out_proj = nn.Linear(d_model, d_model)

    # 把总维度 D 拆成 num_heads 个 head_dim。
    def split_heads(self, x):
        # Tensor.shape 返回各维长度；这里 x 的形状是 [B,L,D]。
        batch_size, seq_len, _ = x.shape
        # view 改变形状但不改变数据：[B,L,D] -> [B,L,H,Dh]。
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # transpose 交换第 1、2 维：[B,L,H,Dh] -> [B,H,L,Dh]。
        return x.transpose(1, 2)

    # 把多个注意力头重新拼回总维度 D。
    def combine_heads(self, x):
        # 此时 x 的形状是 [B,H,L,Dh]。
        batch_size, _, seq_len, _ = x.shape
        # transpose 先得到 [B,L,H,Dh]；contiguous 让内存连续，便于 view。
        x = x.transpose(1, 2).contiguous()
        # 合并最后两个维度：[B,L,H,Dh] -> [B,L,D]。
        return x.view(batch_size, seq_len, self.d_model)

    # forward 定义前向传播；这里实现的是 Q=K=V=x 的 self-attention。
    def forward(self, x, mask=None):
        # 先线性映射，再拆成多个头：[B,L,D] -> [B,H,L,Dh]。
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        # transpose(-2,-1) 交换 K 的最后两维；@ 是矩阵乘法运算符。
        scores = q @ k.transpose(-2, -1)
        # 除以 sqrt(Dh)，避免点积过大导致 softmax 梯度过小。
        scores = scores / math.sqrt(self.head_dim)

        # if 判断 mask 是否存在；is not None 是 Python 的空值判断。
        if mask is not None:
            # mask==0 的位置不可见，用极小值替换，使其 softmax 后接近 0。
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax 在最后一维归一化，使每个 query 对所有 key 的权重之和为 1。
        weights = torch.softmax(scores, dim=-1)
        # 注意力权重乘 V：[B,H,L,L] @ [B,H,L,Dh] -> [B,H,L,Dh]。
        output = weights @ v

        # 合并所有头：[B,H,L,Dh] -> [B,L,D]。
        output = self.combine_heads(output)
        # 输出投影进一步融合不同注意力头的信息。
        output = self.out_proj(output)
        # return 可以一次返回多个对象，这里同时返回输出和注意力权重。
        return output, weights





# __name__ 是 Python 内置变量；直接运行本文件时它等于 "__main__"。
if __name__ == "__main__":
    # randn 创建标准正态随机张量；这里 B=2、L=4、D=8。
    x = torch.randn(2, 4, 8)
    # ones 创建全 1 矩阵，tril 保留下三角；None 用于增加可广播的维度。
    causal_mask = torch.tril(torch.ones(4, 4))[None, None, :, :]

    # 实例化类；D=8，H=2，所以每个头的维度 Dh=4。
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    # 调用模块等价于执行 mha.forward；元组解包得到两个返回值。
    output, weights = mha(x, causal_mask)
    # print 输出结果；.shape 查看张量形状。
    print("output shape:", output.shape)       # [2,4,8]
    # 权重中包含 batch、head、query 和 key 四个维度。
    print("weights shape:", weights.shape)     # [2,2,4,4]

