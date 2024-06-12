import os
import torch
import argparse
import torch.nn as nn

from utils import config

from output_block import OutputBlock
from residual_in_residual import ResidualInResidual

from torchview import draw_graph
from torchsummary import summary


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.stride_size = 1
        self.padding_size = 1

        self.layers = []

        self.input_block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=True,
        )

        self.residual_in_residual_denseblock = nn.Sequential(
            *[ResidualInResidual(in_channels=self.out_channels) for _ in range(3)]
        )

        self.middle_block = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=True,
        )

        self.output = nn.Sequential(
            OutputBlock(in_channels=self.out_channels, out_channels=self.out_channels),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
            ),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            input_block = self.input_block(x)
            residual_block = self.residual_in_residual_denseblock(input_block)
            middle_block = self.middle_block(residual_block)
            middle_block = torch.add(input_block, middle_block)
            output = self.output(middle_block)

            return output

    @staticmethod
    def total_params(model=None):
        if isinstance(model, Generator):
            return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator for ESRGAN".capitalize())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the channels of the image".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Define the channels of the image".capitalize(),
    )

    args = parser.parse_args()

    netG = Generator(in_channels=args.in_channels, out_channels=args.out_channels)

    print(netG(torch.randn(1, 3, 64, 64)).size())

    assert Generator.total_params(model=netG) == 26893315

    assert netG(torch.randn(1, 3, 64, 64)).size() == (1, 3, 256, 256)

    draw_graph(model=netG, input_data=torch.randn(1, 3, 64, 64)).visual_graph.render(
        filename=os.path.join(config()["path"]["ARTIFACTS_PATH"], "netG"), format="png"
    )

    print(summary(model=netG, input_size=(3, 64, 64)))
