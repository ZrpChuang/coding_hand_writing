"""PPO：GAE + 裁剪策略损失 + Value Loss + Entropy Bonus 的逐行注释版。"""

# torch 提供张量运算和自动求导。
import torch


# GAE 用 reward 和 value 估计每一步 advantage；gamma 是折扣率，lam 控制平滑。
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """rewards/dones:[T]；values:[T+1]，最后一个 value 用于 bootstrap。"""
    # zeros_like 创建与 rewards 同形状的全 0 Tensor，用来保存每一步 advantage。
    advantages = torch.zeros_like(rewards)
    # 创建标量 0；device 保证它与 rewards 位于同一个 CPU/GPU。
    gae = torch.tensor(0.0, device=rewards.device)

    # range(T-1,-1,-1) 表示从最后一步倒序遍历到第 0 步。
    for t in range(len(rewards) - 1, -1, -1):
        # done=1 表示 episode 已结束，此时不能使用下一状态的 value。
        not_done = 1.0 - dones[t]
        # TD 残差 delta=r_t+gamma*V(s_{t+1})-V(s_t)。
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        # GAE 递推：累积未来 TD 残差，同时用 gamma*lambda 逐步衰减。
        gae = delta + gamma * lam * not_done * gae
        # 列表下标赋值，把当前结果保存到对应时间步。
        advantages[t] = gae

    # return=advantage+value，因为 A_t=R_t-V(s_t)。
    returns = advantages + values[:-1]
    # 一次 return 两个对象，调用者可以使用元组解包接收。
    return advantages, returns


# entropy 是策略分布的熵；value_coef、entropy_coef 是各损失项权重。
def ppo_loss(new_logp, old_logp, advantages, new_values, returns, entropy,
             clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
    # exp(log pi_new-log pi_old) 得到重要性采样概率比。
    ratio = torch.exp(new_logp - old_logp)
    # 未裁剪的策略目标。
    surrogate1 = ratio * advantages
    # clamp 限制概率比，再乘 advantage 得到裁剪目标。
    surrogate2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    # PPO 最大化两个目标的较小者；训练最小化 loss，所以前面加负号。
    policy_loss = -torch.minimum(surrogate1, surrogate2).mean()

    # **2 表示平方；Value 网络使用均方误差拟合 returns。
    value_loss = ((new_values - returns) ** 2).mean()
    # 熵越大探索越强；最小化 loss 时要减去 entropy bonus。
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
    # 同时返回总损失和两个子损失，方便观察训练日志。
    return total_loss, policy_loss, value_loss


# Python 程序入口保护：直接运行时执行 demo。
if __name__ == "__main__":
    # 构造三步 rollout 的奖励。
    rewards = torch.tensor([1.0, 0.2, 0.5])
    # 四个 value 对应三个当前状态加一个 bootstrap 的下一状态。
    values = torch.tensor([0.8, 0.4, 0.3, 0.0])
    # 最后一步 done=1，表示 episode 在最后一步结束。
    dones = torch.tensor([0.0, 0.0, 1.0])
    # 计算 PPO 训练需要的 advantage 和 return。
    advantages, returns = compute_gae(rewards, values, dones)

    # old_logp 来自 rollout 时冻结的旧策略。
    old_logp = torch.tensor([-0.5, -0.8, -1.0])
    # new_logp 来自当前策略，需要计算梯度，所以 requires_grad=True。
    new_logp = torch.tensor([-0.4, -0.9, -0.7], requires_grad=True)
    # new_values 是当前 Value 网络预测，也需要计算梯度。
    new_values = torch.tensor([0.9, 0.5, 0.2], requires_grad=True)
    # entropy 通常从策略概率分布计算，这里直接构造最小 demo。
    entropy = torch.tensor([0.6, 0.5, 0.7])

    # 调用 PPO loss，并使用元组解包取得三个返回值。
    loss, policy_loss, value_loss = ppo_loss(
        new_logp, old_logp, advantages, new_values, returns, entropy
    )
    # 执行反向传播，计算 new_logp 和 new_values 的梯度。
    loss.backward()
    # item() 把标量 Tensor 转成 Python 浮点数后打印。
    print("total loss:", loss.item())
    # 打印策略裁剪损失。
    print("policy loss:", policy_loss.item())
    # 打印 Value 网络的均方误差。
    print("value loss:", value_loss.item())

