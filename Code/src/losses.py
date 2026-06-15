import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, weight=None, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        targets = targets.long()
        num_classes = inputs.shape[1]
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        targets_safe = torch.where(targets == self.ignore_index, torch.tensor(0, device=targets.device), targets)
        inputs_softmax = F.softmax(inputs, dim=1)

        targets_onehot = F.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()

        inputs_softmax = inputs_softmax * valid_mask
        targets_onehot = targets_onehot * valid_mask

        intersection = (inputs_softmax * targets_onehot).sum(dim=(2, 3))
        cardinality = (inputs_softmax + targets_onehot).sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        if self.weight is not None:
            dice_score = dice_score * self.weight.to(dice_score.device)

        return 1 - dice_score.mean()


class CombinedLoss(nn.Module):

    def __init__(self, dice_weight=0.5, lambda_phy=0.1, ignore_index=255):
        super().__init__()
        self.dice_weight = dice_weight
        self.lambda_phy = lambda_phy
        self.ignore_index = ignore_index

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.phy_ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, logits, targets, phy_pred=None, phy_gt=None):

        loss_ce = self.ce_loss(logits, targets.long())
        loss_dice = self.dice_loss(logits, targets.long())

        semantic_loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        if phy_pred is not None and phy_gt is not None:
            loss_smpc = self.phy_ce_loss(phy_pred, phy_gt.long())

            total_loss = semantic_loss + self.lambda_phy * loss_smpc
            return total_loss, semantic_loss, loss_smpc

        return semantic_loss, semantic_loss, torch.tensor(0.0).to(logits.device)