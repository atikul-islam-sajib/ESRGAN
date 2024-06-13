import os
import sys
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("/src/")

from utils import config


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 3
        self.stride_size = 2
        self.padding_size = 1
        self.negative_slope = 0.2

        self.layers = []

        self.input_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
        )

        for idx in range(3):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels * 2,
                        kernel_size=self.kernel_size,
                        stride=(
                            self.stride_size // 2 if idx % 3 == 0 else self.stride_size
                        ),
                        padding=self.padding_size,
                    ),
                    nn.BatchNorm2d(num_features=self.out_channels * 2),
                    nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
                )
            )
            self.out_channels = self.out_channels * 2

        self.immediate_block = nn.Sequential(*self.layers)

        self.ouput_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.in_channels // self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size // self.stride_size,
            )
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            input = self.input_block(x)
            immediate = self.immediate_block(input)
            output = self.ouput_block(immediate)

            return output

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, Discriminator):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        else:
            raise TypeError("Model must be a Discriminator".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discriminator block for ESRGAN".title()
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Input channels".capitalize()
    )
    parser.add_argument(
        "--out_channels", type=int, default=64, help="Output channels".capitalize()
    )

    args = parser.parse_args()

    netD = Discriminator(in_channels=args.in_channels, out_channels=args.out_channels)

    assert netD(torch.randn(1, 3, 256, 256)).size() == (1, 1, 30, 30)

    assert Discriminator.total_params(netD) == 1557377

    draw_graph(model=netD, input_data=torch.randn(1, 3, 256, 256)).visual_graph.render(
        filename=os.path.join(config()["path"]["ARTIFACTS_PATH"], "netD"), format="png"
    )

    print(summary(model=netD, input_size=(3, 256, 256)))
