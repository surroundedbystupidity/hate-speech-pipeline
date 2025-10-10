import torch
from torch import nn


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Give more weight to positive (toxic) examples
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor([self.pos_weight]).to(logits.device),
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce_loss).mean()


class HybridLoss(nn.Module):
    def __init__(self, mse_weight=0.5, margin=0.9):
        super().__init__()
        self.mse_weight = mse_weight
        self.margin = margin

    def forward(self, predictions, targets):
        # MSE component
        mse_loss = nn.functional.mse_loss(predictions, targets)

        # Ranking component - ensure toxic > non-toxic predictions
        toxic_mask = targets > self.mse_weight
        non_toxic_mask = targets <= self.mse_weight

        if toxic_mask.any() and non_toxic_mask.any():
            toxic_preds = predictions[toxic_mask]
            non_toxic_preds = predictions[non_toxic_mask]

            # Pairwise ranking loss
            diff = toxic_preds.unsqueeze(1) - non_toxic_preds.unsqueeze(0)
            ranking_loss = torch.relu(self.margin - diff).mean()
        else:
            ranking_loss = 0

        return self.mse_weight * mse_loss + (1 - self.mse_weight) * ranking_loss
