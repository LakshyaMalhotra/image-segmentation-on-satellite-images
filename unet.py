"""Implementing U-Net from scratch.
"""

import torch
import torch.nn as nn


def down_conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
    )

    return block


def crop_connection(input_tensor, target_tensor):
    target_size = target_tensor.size()[-1]
    input_size = input_tensor.size()[-1]
    delta = input_size - target_size
    delta = delta // 2
    return input_tensor[
        :,
        :,
        delta : input_size - delta,
        delta : input_size - delta,
    ]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # encoder
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block1 = down_conv_block(in_channels=1, out_channels=64)
        self.down_block2 = down_conv_block(in_channels=64, out_channels=128)
        self.down_block3 = down_conv_block(in_channels=128, out_channels=256)
        self.down_block4 = down_conv_block(in_channels=256, out_channels=512)
        self.down_block5 = down_conv_block(in_channels=512, out_channels=1024)

        # decoder
        self.up_conv4 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_block4 = down_conv_block(in_channels=1024, out_channels=512)

        self.up_conv3 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_block3 = down_conv_block(in_channels=512, out_channels=256)

        self.up_conv2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_block2 = down_conv_block(in_channels=256, out_channels=128)

        self.up_conv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_block1 = down_conv_block(in_channels=128, out_channels=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down_block1(x)
        print(x1.size())
        x1_down = self.max_pool(x1)

        x2 = self.down_block2(x1_down)
        print(x2.size())
        x2_down = self.max_pool(x2)

        x3 = self.down_block3(x2_down)
        print(x3.size())
        x3_down = self.max_pool(x3)

        x4 = self.down_block4(x3_down)
        print(x4.size())
        x4_down = self.max_pool(x4)

        x5 = self.down_block5(x4_down)
        print(x5.size())

        # decoder
        x_up = self.up_conv4(x5)
        print(x_up.size())
        x4_connection = crop_connection(x4, x_up)
        x_up = self.up_block4(torch.cat([x4_connection, x_up], dim=1))
        print(x_up.size())

        x_up = self.up_conv3(x_up)
        print(x_up.size())
        x3_connection = crop_connection(x3, x_up)
        x_up = self.up_block3(torch.cat([x3_connection, x_up], dim=1))
        print(x_up.size())

        x_up = self.up_conv2(x_up)
        print(x_up.size())
        x2_connection = crop_connection(x2, x_up)
        x_up = self.up_block2(torch.cat([x2_connection, x_up], dim=1))
        print(x_up.size())

        x_up = self.up_conv1(x_up)
        print(x_up.size())
        x1_connection = crop_connection(x1, x_up)
        x_up = self.up_block1(torch.cat([x1_connection, x_up], dim=1))
        print(x_up.size())

        x_up = self.conv_1x1(x_up)
        print(x_up.size())


if __name__ == "__main__":
    image = torch.rand(1, 1, 572, 572)
    model = UNet()
    y = model(image)
