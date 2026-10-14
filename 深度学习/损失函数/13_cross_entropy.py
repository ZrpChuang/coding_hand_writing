"""Cross Entropy：不用 F.cross_entropy 的手撕版。"""

# torch 提供 logsumexp、gather 和 Tensor 运算。
import torch


# logits:[B,C] 是未归一化分类分数；targets:[B] 是正确类别下标。
def cross_entropy(logits, targets):
    # log_softmax(x_i)=x_i-log(sum(exp(x)))；logsumexp 写法数值稳定。
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # unsqueeze(-1) 把 [B] 变为 [B,1]，满足 gather 的索引形状要求。
    target_index = targets.unsqueeze(-1)
    # gather 沿类别维取出每个样本正确类别的 log probability。
    target_log_probs = log_probs.gather(dim=-1, index=target_index)
    # squeeze(-1) 删除大小为 1 的类别维：[B,1] -> [B]。
    target_log_probs = target_log_probs.squeeze(-1)
    # 交叉熵是正确类别 log probability 的负均值。
    return -target_log_probs.mean()


# 直接运行本文件时执行 demo。
if __name__ == "__main__":
    # 两个样本、三个类别；requires_grad=True 表示需要对 logits 求梯度。
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.2, 0.3, 1.5]], requires_grad=True)
    # 第一个样本正确类别是 0，第二个样本正确类别是 2。
    targets = torch.tensor([0, 2])
    # 调用手写函数计算一个标量 loss。
    loss = cross_entropy(logits, targets)
    # 反向传播计算 logits.grad。
    loss.backward()
    # 打印交叉熵数值。
    print("cross entropy:", loss.item())

