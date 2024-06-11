import torch
import argparse
import torch.nn as nn

from dense_block import DenseBlock


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

        for _ in range(5):
            self.layers.append(
                DenseBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                )
            )

        self.residual_in_residual_denseblock = nn.Sequential(*self.layers)

        self.middle_block = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=True,
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            input_block = self.input_block(x)
            residual_block = self.residual_in_residual_denseblock(input_block)
            middle_block = self.middle_block(residual_block)
            middle_block = torch.add(input_block, middle_block)

            return middle_block


if __name__ == "__main__":
    netG = Generator(in_channels=3, out_channels=64)

    print(netG(torch.randn(1, 3, 256, 256)).size())
