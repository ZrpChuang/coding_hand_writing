"""GRPO：组内奖励标准化 + PPO 裁剪 + KL 约束的逐行注释版。"""

# torch 提供张量计算、自动求导以及常用数学函数。
import torch


# def 定义函数；eps 是防止除以 0 的极小数，调用时可以不传。
def group_advantages(rewards, eps=1e-8):
    """rewards: [B,G]；B 个 prompt，每个 prompt 生成 G 个回答。"""
    # dim=1 表示在每组 G 个回答内求均值；keepdim 保留维度以便广播。
    mean = rewards.mean(dim=1, keepdim=True)

    # std 求标准差；unbiased=False 使用总体标准差，避免小组上的无偏修正。
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    # 高于组内平均奖励得到正优势，低于平均奖励得到负优势。
    return (rewards - mean) / (std + eps)


# new/old/ref_logp 都是已生成 token 的 log probability。
def grpo_loss(new_logp, old_logp, ref_logp, advantages, mask,
              clip_eps=0.2, beta=0.01):
    """logp/mask:[B,G,L]；advantages:[B,G]；mask 中 1 是有效 token。"""
    # unsqueeze(-1) 在末尾增加 token 维：[B,G] -> [B,G,1]，之后自动广播。
    advantages = advantages.unsqueeze(-1)

    # log(a/b)=log(a)-log(b)，所以 exp(logp_new-logp_old)=pi_new/pi_old。
    ratio = torch.exp(new_logp - old_logp)
    # clamp 把概率比限制在 [1-epsilon,1+epsilon]，防止策略更新过猛。
    clipped_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    # minimum 实现 PPO clipped objective；正负 advantage 都能正确裁剪。
    policy_gain = torch.minimum(
        ratio * advantages,
        clipped_ratio * advantages,
    )

    # 先计算 log(pi_ref/pi_new)，后面用于近似逐 token KL。
    log_ratio = ref_logp - new_logp
    # exp(x)-x-1 恒为非负，是 KL(pi_new || pi_ref) 的逐样本估计。
    kl = torch.exp(log_ratio) - log_ratio - 1
    # 为什么是指数形式：exp(x) ≈ 1 + x + x²/2，所以 exp(x)-x-1 的二次项 ≈ x²/2，就是 KL 的二阶近似。

    # 最大化策略收益、惩罚偏离参考模型；beta 控制 KL 约束强度。
    token_objective = policy_gain - beta * kl
    # 先对每条回答的有效 token 求平均，避免长回答获得更大权重。
    sequence_objective = (
        (token_objective * mask).sum(dim=-1)
        / mask.sum(dim=-1).clamp(min=1)
    )
    # 优化器默认最小化 loss，因此对需要最大化的 objective 取负号。
    return -sequence_objective.mean()


# 直接运行本文件时执行 demo；被 import 时不执行。
if __name__ == "__main__":
    # manual_seed 固定随机种子，使每次运行结果一致。
    torch.manual_seed(0)

    # 两个 prompt，每个 prompt 有三个回答的奖励，所以形状是 [B=2,G=3]。
    rewards = torch.tensor([[1.0, 0.2, 0.5], [0.1, 0.9, 0.4]])

    # 通过组内标准化得到每个回答的 advantage。
    advantages = group_advantages(rewards)

    # randn 构造旧策略和参考策略的 token log probability：[B,G,L]。
    old_logp = torch.randn(2, 3, 4)
    ref_logp = torch.randn(2, 3, 4)
    # requires_grad_() 原地开启梯度记录，模拟当前待训练策略的输出。
    new_logp = (old_logp + 0.1 * torch.randn(2, 3, 4)).requires_grad_()
    # ones_like 创建同形状全 1 张量，表示 demo 中所有 token 都有效。
    mask = torch.ones_like(new_logp)

    # 调用函数计算一个标量 GRPO loss。
    loss = grpo_loss(new_logp, old_logp, ref_logp, advantages, mask)
    # backward 根据计算图反向传播，把梯度写入 new_logp.grad。
    loss.backward()
    # \n 是换行转义字符；print 用来查看组内优势。
    print("advantages:\n", advantages)
    # item() 把只含一个元素的 Tensor 转成普通 Python 数字。
    print("GRPO loss:", loss.item())





