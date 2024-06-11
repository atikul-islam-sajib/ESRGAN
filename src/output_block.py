import torch
import argparse
import torch.nn as nn


class OutputBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(OutputBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 3
        self.stride_size = 1
        self.padding_size = 1
        self.negative_slope = 0.2
        self.upscale_factor = 2

        self.output_block = self.block()

    def block(self):

        self.layers = []

        for idx in range(2):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.in_channels * 4,
                        kernel_size=self.kernel_size,
                        stride=self.stride_size,
                        padding=self.padding_size,
                        bias=True,
                    ),
                    nn.PixelShuffle(upscale_factor=self.upscale_factor),
                )
            )
            if idx == 0:
                self.layers.append(
                    nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
                )

        return nn.Sequential(*self.layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.output_block(x)
        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    outblock = OutputBlock(in_channels=64, out_channels=3)

    print(outblock(torch.randn(1, 64, 64, 64)).size())
