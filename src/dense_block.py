import torch
import argparse
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.slope = 0.2

        self.block1 = self.block(
            in_channels=1 * self.in_channels,
            out_channels=self.out_channels,
            use_leaky=True,
        )
        self.block2 = self.block(
            in_channels=2 * self.in_channels,
            out_channels=self.out_channels,
            use_leaky=True,
        )
        self.block3 = self.block(
            in_channels=3 * self.in_channels,
            out_channels=self.out_channels,
            use_leaky=True,
        )
        self.block4 = self.block(
            in_channels=4 * self.in_channels,
            out_channels=self.out_channels,
            use_leaky=True,
        )
        self.block5 = self.block(
            in_channels=5 * self.in_channels,
            out_channels=self.out_channels,
            use_leaky=False,
        )

    def block(self, in_channels=64, out_channels=64, use_leaky=True):
        self.layers = []

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=True,
            )
        )

        if use_leaky:
            self.layers.append(nn.LeakyReLU(negative_slope=self.slope, inplace=True))

        return nn.Sequential(*self.layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            outputs = self.block1(x)
            inputs = torch.concat((outputs, x), dim=1)

            outputs = self.block2(inputs)
            inputs = torch.concat((outputs, inputs), dim=1)

            outputs = self.block3(inputs)
            inputs = torch.concat((outputs, inputs), dim=1)

            outputs = self.block4(inputs)
            inputs = torch.concat((outputs, inputs), dim=1)

            outputs = self.block5(inputs)

            return outputs

        else:
            raise TypeError("Input must be a tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseBlock for netG".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Define the in_channels of the DenseBlock".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Define the out_channels of the DenseBlock".capitalize(),
    )
    args = parser.parse_args()

    layers = []

    for _ in range(5):
        layers += [
            DenseBlock(in_channels=args.in_channels, out_channels=args.out_channels)
        ]

    model = nn.Sequential(*layers)

    assert model(torch.randn(1, 64, 256, 256)).size() == (1, 64, 256, 256)
