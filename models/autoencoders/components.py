import torch
import torch.nn as nn
from collections import OrderedDict

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
        if features_maps is not None:
            x = torch.cat(self.crop(features_maps, x), 1)
        x = self.conv_blocks(x)
        return x

    def crop(self, x, y):
        final_shape = [min(x.shape[i], y.shape[i]) for i in range(2,4)]
        return [t[:, :, :final_shape[-2], :final_shape[-1]] for t in (x, y)]

class Bridge(nn.Module):
    def __init__(self, input_dim, output_dims):
        super().__init__()

        layers = []
        for output_dim in output_dims:
            layers.append(ConvBlock(input_dim, output_dim))
            input_dim = output_dim

        self.bridge = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.bridge(x)

    @property
    def in_channels(self):
        return self.bridge[0].conv.in_channels
    
    @property
    def out_channels(self):
        return self.bridge[-1].conv.out_channels

class UnetDecoder(nn.Module):
    def __init__(self, 
        decoder_channels,
        upsampling_method,
        output_size,
        copy_n_crop=True,
        out_channels=3 
    ):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.output_size = output_size
        self.upsampling_method = upsampling_method
        self.copy_n_crop = copy_n_crop
        self.up_blocks = self.build_up_blocks()
        
        self.last_layer = nn.Conv2d(
            in_channels=decoder_channels[-1][1], 
            out_channels=out_channels,
            kernel_size=1, stride=1
        )
        
        

    def build_up_blocks(self):
        up_blocks = {}
        for i, o in self.decoder_channels:
            up_blocks['{};{}'.format(i, o)] = (
                UpBlock(
                    i, o, o*(2 if self.copy_n_crop else 1), o,
                    self.upsampling_method
                )
            )
        return nn.ModuleDict(up_blocks)

    def forward(self, x, encoder_features_maps=None):
        if encoder_features_maps is None:
            encoder_features_maps = [None] * len(self.up_blocks)

        for block, fm in zip(self.up_blocks.values(), encoder_features_maps):
            x = block(x, fm)
        x = self.last_layer(x)
        x = nn.functional.interpolate(x, self.output_size)
        return x

class ResnetEncoder(nn.Module):
    
    def __init__(self, resnet):
        super().__init__()
        self.layers = list(resnet.children())
        
        self.input_block = nn.Sequential(*self.layers[:3])
        self.max_pool = self.layers[3]
        self.conv_blocks, self.conv_block_channel_shapes = self.build_conv_blocks()
        # self.conv_block_channel_shapes.insert(0, self.get_layer_channels(self.input_block[0]))

    def build_conv_blocks(self):
        block_channel_shapes = []
        blocks = {}
        for l in self.layers:
            if isinstance(l, nn.Sequential):
                channel_shapes = self.get_layer_channels(l[0].conv1) 
                block_channel_shapes.append(channel_shapes)
                blocks[self.compute_block_name(l[0].conv1)] = l
            
        return (nn.ModuleDict(blocks), block_channel_shapes)

    def compute_block_name(self, block):
        channel_shapes = (block.in_channels, block.out_channels)
        return '{};{}'.format(*channel_shapes)

    def get_layer_channels(self, layer):
        return (layer.in_channels, layer.out_channels)

    def forward(self, x):
        features_maps = OrderedDict()
        x = self.input_block(x)

        features_maps[self.compute_block_name(self.input_block[0])] = x

        x = self.max_pool(x)
        for name, conv_block in self.conv_blocks.items():
            x = conv_block(x)
            features_maps[name] = x
        return features_maps, x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def last_layer_out_channels(self):
        return list(self.conv_blocks.values())[-1][-1].conv2.out_channels 