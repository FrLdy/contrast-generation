import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    UPSAMPLING_BILINEAR = "bilinear"
    UPSAMPLING_CONV_TRANSPOSE = "conv_transpose"

    def __init__(
        self, 
        up_in_channels, up_out_channels,
        conv_in_channels, conv_out_channels,
        upsampling_method="bilinear"
    ):
        super().__init__()
         
        self.upsample = self.get_up_sampling_method(upsampling_method)(
            up_in_channels, 
            up_out_channels
        )

        self.conv_blocks = nn.Sequential(
            ConvBlock(conv_in_channels, conv_out_channels),
            ConvBlock(conv_out_channels, conv_out_channels)
        )
    
    def get_up_sampling_method(self, method_name):
        if method_name == self.UPSAMPLING_BILINEAR :
            return self.bilinear_upsampling
        elif method_name == self.UPSAMPLING_CONV_TRANSPOSE :
            return self.conv_transpose_upsampling

    def conv_transpose_upsampling(self, *args):
        return nn.Sequential(
            nn.ConvTranspose2d(
                *args,
                kernel_size=2, stride=2
            )
        )

    def bilinear_upsampling(self, *args):
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
            nn.Conv2d(
                *args, 
                kernel_size=1, stride=1
            )
        )
    
    def forward(self, x, features_maps=None):
        x = self.upsample(x)
        if features_maps is not None :
            x = torch.cat(self.crop(x, features_maps), 1)
        x = self.conv_blocks(x)
        return x

    def crop(self, x, y):
        final_shape = [min(x.shape[i], y.shape[i]) for i in range(2,4)]
        return [t[:, :, :final_shape[-2], :final_shape[-2]] for t in (x, y)]