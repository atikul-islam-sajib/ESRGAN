import torch
import argparse
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()

        self.reduction = reduction

        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            return self.loss(pred, target)

        else:
            raise TypeError("Input must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSELoss")
    parser.add_argument(
        "--reduction", type=str, default="mean", help="reduction method"
    )
    args = parser.parse_args()

    loss = MSELoss(args.reduction)

    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    predicted = torch.tensor([1.0, 0.0, 1.0, 1.0])

    print(loss(predicted, target))
