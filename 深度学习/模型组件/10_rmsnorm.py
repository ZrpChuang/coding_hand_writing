"""RMSNorm：只按均方根缩放，不减均值。"""

# torch 提供张量运算和 rsqrt。
import torch
# nn 提供 Module 和可训练参数 Parameter。
from torch import nn


# RMSNorm 继承 nn.Module，因此可以放入神经网络并参与训练。
class RMSNorm(nn.Module):
    # dim 是输入最后一维大小；eps 防止除以 0。
    def __init__(self, dim, eps=1e-6):
        # 调用父类构造函数。
        super().__init__()
        # nn.Parameter 把全 1 Tensor 注册为可训练缩放参数 gamma。
        self.weight = nn.Parameter(torch.ones(dim))
        # 保存极小数 eps。
        self.eps = eps

    # forward 定义归一化操作，x 通常形如 [B,L,D]。
    def forward(self, x):
        # pow(2) 逐元素平方；mean(-1) 在特征维求均值，并保留该维。
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt(z)=1/sqrt(z)，将 x 除以其均方根。
        normalized = x * torch.rsqrt(mean_square + self.eps)
        # 乘可训练参数 weight；PyTorch 会自动沿 B、L 维广播。
        return self.weight * normalized


# 直接运行文件时执行 demo。
if __name__ == "__main__":
    # 创建形状 [B=2,L=3,D=4] 的随机输入。
    x = torch.randn(2, 3, 4)
    # 创建特征维为 4 的 RMSNorm 层。
    norm = RMSNorm(dim=4)
    # 调用模块执行 forward。
    y = norm(x)
    # RMSNorm 不改变张量形状。
    print("output shape:", y.shape)             # [2,3,4]

