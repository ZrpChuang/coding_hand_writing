"""DPO Loss：让策略更偏好 chosen、远离 rejected 的逐行注释版。"""

# torch 提供 Tensor 和自动求导。
import torch
# F 是 torch.nn.functional 的常用别名，这里使用数值稳定的 logsigmoid。
import torch.nn.functional as F


# 四个 logp 都是整条回答所有有效 token 的 log probability 之和。
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    # 当前策略对 chosen 相比 rejected 的偏好差值。
    policy_gap = policy_chosen_logp - policy_rejected_logp
    # 参考策略本身也可能偏好 chosen，因此需要减去参考策略的偏好差值。
    ref_gap = ref_chosen_logp - ref_rejected_logp
    # beta 控制偏好优化强度；差值越大，说明当前策略比参考策略更偏好 chosen。
    logits = beta * (policy_gap - ref_gap)
    # -log(sigmoid(logits)) 是二分类负对数似然；logsigmoid 写法更数值稳定。
    return -F.logsigmoid(logits).mean()


# 直接运行本文件时执行 demo，被其他模块导入时不会执行。
if __name__ == "__main__":
    # 当前策略 chosen 回答的序列 logp；requires_grad=True 表示参与训练。
    policy_chosen = torch.tensor([-2.0, -1.5], requires_grad=True)
    # 当前策略 rejected 回答的序列 logp，也需要计算梯度。
    policy_rejected = torch.tensor([-3.0, -2.2], requires_grad=True)
    # 参考策略保持冻结，因此默认不需要梯度。
    ref_chosen = torch.tensor([-2.2, -1.7])
    # 参考策略对 rejected 回答的序列 logp。
    ref_rejected = torch.tensor([-2.8, -2.0])

    # 调用函数得到 batch 平均后的标量 DPO loss。
    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    # 反向传播后，chosen 的概率会被推高，rejected 的概率会被压低。
    loss.backward()
    # item() 把标量 Tensor 转成 Python 数字后打印。
    print("DPO loss:", loss.item())

