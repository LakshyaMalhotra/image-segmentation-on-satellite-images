import torch
import torch.nn as nn
import torch.nn.functional as F


class OverallLoss(nn.Module):
    """Helper class to calculate the total loss function. The overall loss is
    defined as the sum of the pixel-wise binary cross entropy and dice loss both
    scaled by the fraction showing the relative weights of both losses.
    """

    def __init__(self, bce_weight: float = 0.4):
        super(OverallLoss, self).__init__()
        self.bce_weight = bce_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, metrics: dict
    ) -> torch.Tensor:
        """Find the total loss by adding BCE to the dice loss.

        Args:
        -----
            pred (torch.Tensor): Model output (logits).
            target (torch.Tensor): Ground truth segmentation mask.
            metrics (dict): Dictionary to hold different losses.

        Returns:
        --------
            torch.Tensor: Overall loss.
        """
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred)
        dice = self.dice_loss(pred, target)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)

        metrics["bce"] += bce.detach().cpu().numpy()
        metrics["dice"] += dice.detach().cpu().numpy()
        metrics["total"] += loss.detach().cpu().numpy()

        return loss

    @staticmethod
    def dice_loss(
        pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
    ) -> float:
        """Calculates the dice loss
        dice loss = 1 - dice coefficient
        Implementation borrowed from: https://github.com/usuyama/pytorch-unet/blob/master/loss.py

        Args:
        -----
            pred (torch.Tensor): Model output (probabilities).
            target (torch.Tensor): Ground truth segmentation mask.
            smooth (float, optional): Smoothening condition. Defaults to 1.0.

        Returns:
        --------
            float: Dice loss
        """
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = 1 - (
            (2.0 * intersection + smooth)
            / (
                pred.sum(dim=2).sum(dim=2)
                + target.sum(dim=2).sum(dim=2)
                + smooth
            )
        )

        return loss.mean()

    @staticmethod
    def print_metrics(metrics: dict, n_batches: int, phase: str) -> None:
        """Function to print the metrics during training/validation. Prints the
        batch average of various losses.

        Args:
        -----
            metrics (dict): Dictionary to hold different losses.
            n_batches (int): Number of batches in dataloader.
            phase (str): Training or validation.
        """
        outputs = []
        for k in metrics.keys():
            outputs.append(f"{k} loss: {(metrics[k] / n_batches):.5f}")

        print(f"Phase: {phase}: {', '.join(outputs)}")
