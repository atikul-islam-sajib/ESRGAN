import torch
import argparse
import torch.nn as nn
import torchvision.models as models

import warnings

warnings.filterwarnings("ignore")


class VGG19(nn.Module):
    def __init__(self, name="VGG19"):
        super(VGG19, self).__init__()

        self.name = name

        self.model = models.vgg19(pretrained=True)

        for params in self.model.parameters():
            params.requires_grad = False

        self.model = nn.Sequential(*list(self.model.children())[0][:35])

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.model(x)
        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG19".title())
    parser.add_argument(
        "--name", type=str, default="VGG19", help="Name of the model".capitalize()
    )

    args = parser.parse_args()

    model = VGG19(name=args.name)

    assert model(torch.randn(1, 3, 256, 256)).size() == (1, 512, 16, 16)
