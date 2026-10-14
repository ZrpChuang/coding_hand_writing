"""真实 DPO forward：模型输入 -> logits -> 序列 logp -> DPO loss。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_logp(logits, input_ids, loss_mask):
    """计算每条回答的 log probability 之和。

    logits:    [B, L, V]
    input_ids: [B, L]
    loss_mask: [B, L]，回答 token 为 1，prompt/padding 为 0
    """
    # 位置 t 的 logits 预测位置 t+1 的 token。
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    loss_mask = loss_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    return (token_logp * loss_mask).sum(dim=-1)


class DPOModel(nn.Module):
    """封装可训练策略模型和冻结参考模型的 DPO forward。"""

    def __init__(self, policy_model, ref_model, beta=0.1):
        super().__init__()
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta

        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

    def train(self, mode=True):
        """训练 policy 时始终让 reference 保持评估模式。"""
        super().train(mode)
        self.ref_model.eval()
        return self

    def forward(
        self,
        chosen_input_ids,
        rejected_input_ids,
        chosen_attention_mask,
        rejected_attention_mask,
        chosen_loss_mask,
        rejected_loss_mask,
    ):
        # Policy 模型需要梯度。
        policy_chosen_logits = self.policy_model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
        ).logits
        policy_rejected_logits = self.policy_model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
        ).logits

        # Reference 模型只提供基准概率，不参与训练。
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
            ).logits
            ref_rejected_logits = self.ref_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
            ).logits

        policy_chosen_logp = sequence_logp(
            policy_chosen_logits, chosen_input_ids, chosen_loss_mask
        )
        policy_rejected_logp = sequence_logp(
            policy_rejected_logits, rejected_input_ids, rejected_loss_mask
        )
        ref_chosen_logp = sequence_logp(
            ref_chosen_logits, chosen_input_ids, chosen_loss_mask
        )
        ref_rejected_logp = sequence_logp(
            ref_rejected_logits, rejected_input_ids, rejected_loss_mask
        )

        policy_gap = policy_chosen_logp - policy_rejected_logp
        ref_gap = ref_chosen_logp - ref_rejected_logp
        dpo_logits = self.beta * (policy_gap - ref_gap)

        return -F.logsigmoid(dpo_logits).mean()
