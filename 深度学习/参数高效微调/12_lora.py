"""LoRA：冻结原线性层，只训练低秩增量 BA。"""

# torch 提供 Tensor 和初始化函数。
import torch
# nn 提供 Linear、Module 等组件。
from torch import nn


# LoRALinear 用低秩分支包装一个普通线性层。
class LoRALinear(nn.Module):
    # rank 是低秩维度；alpha 控制 LoRA 分支的缩放大小。
    def __init__(self, in_features, out_features, rank=4, alpha=8.0):
        # 初始化父类 nn.Module。
        super().__init__()
        # base 是原始线性层，实际微调时通常来自预训练模型。
        self.base = nn.Linear(in_features, out_features)
        # A 先把输入从大维度压缩到 rank。
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        # B 再把 rank 映射到输出维度。
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        # LoRA 论文使用 alpha/rank 缩放低秩更新。
        self.scaling = alpha / rank

        # parameters() 返回 base 的所有参数。
        for parameter in self.base.parameters():
            # requires_grad=False 冻结原模型参数，优化器不会更新它们。
            parameter.requires_grad = False

        # normal_ 原地用正态分布初始化 A。
        nn.init.normal_(self.lora_a.weight, std=0.02)
        # zeros_ 把 B 初始化为 0，使训练开始时 LoRA 分支不改变原模型输出。
        nn.init.zeros_(self.lora_b.weight)

    # forward 实现 W0*x + alpha/r * B*A*x。
    def forward(self, x):
        # base(x) 是冻结预训练权重产生的原始输出。
        base_output = self.base(x)
        # 函数从内到外调用：先经过 A，再经过 B，最后乘缩放系数。
        lora_output = self.lora_b(self.lora_a(x)) * self.scaling
        # + 表示逐元素相加，把低秩增量加入原始输出。
        return base_output + lora_output


# 直接运行本文件时执行 demo。
if __name__ == "__main__":
    # 创建 3 个样本，每个输入维度为 8。
    x = torch.randn(3, 8)
    # 创建 8->4 的线性层，LoRA rank 为 2。
    layer = LoRALinear(in_features=8, out_features=4, rank=2, alpha=4.0)
    # 调用模块执行前向传播。
    y = layer(x)
    # 输出最后一维等于 out_features=4。
    print("output shape:", y.shape)             # [3,4]

