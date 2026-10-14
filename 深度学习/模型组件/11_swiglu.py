"""SwiGLU：现代 LLM 中常用的门控前馈网络。"""

# torch 提供随机 Tensor。
import torch
# nn 提供 Linear 和 Module；F 提供 silu 激活函数。
from torch import nn
# functional 中的函数没有额外参数，适合直接调用。
import torch.nn.functional as F


# SwiGLU 继承 nn.Module。
class SwiGLU(nn.Module):
    # d_model 是输入输出维度；d_ff 是中间层维度。
    def __init__(self, d_model, d_ff):
        # 初始化父类。
        super().__init__()
        # gate_proj 产生门控分支。
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        # up_proj 产生内容分支。
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        # down_proj 把中间维度映射回模型维度。
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    # forward 定义 SwiGLU(x)=down(silu(gate(x))*up(x))。
    def forward(self, x):
        # silu(z)=z*sigmoid(z)，决定哪些中间特征可以通过。
        gate = F.silu(self.gate_proj(x))
        # 内容分支保存需要被门控的信息。
        value = self.up_proj(x)
        # * 是逐元素乘法；门控后再投影回 d_model。
        return self.down_proj(gate * value)


# 直接运行本文件时执行 demo。
if __name__ == "__main__":
    # 创建 [B=2,L=3,D=8] 的输入。
    x = torch.randn(2, 3, 8)
    # 中间维度设为 16，输入输出维度保持 8。
    ffn = SwiGLU(d_model=8, d_ff=16)
    # 执行前向传播。
    y = ffn(x)
    # SwiGLU 最终恢复到模型维度。
    print("output shape:", y.shape)             # [2,3,8]

