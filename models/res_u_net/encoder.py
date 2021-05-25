import torch.nn as nn
from collections import OrderedDict


class ResnetEncoder(nn.Module):
    
    def __init__(self, resnet):
        super().__init__()
        self.layers = list(resnet.children())
        
        self.input_block = nn.Sequential(*self.layers[:3])
        self.conv_blocks, self.conv_block_channel_shapes = self.build_conv_blocks()

    def build_conv_blocks(self):
        block_channel_shapes = []
        blocks = {}
        for l in self.layers:
            if isinstance(l, nn.Sequential):
                channel_shapes = (l[0].conv1.in_channels, l[0].conv1.out_channels)
                block_channel_shapes.append(channel_shapes)
                blocks['{};{}'.format(*channel_shapes)] = l
            
        return (nn.ModuleDict(blocks), block_channel_shapes)

    def forward(self, x):
        x = self.input_block(x)
        features_maps = OrderedDict()
        for name, conv_block in self.conv_blocks.items():
            x = conv_block(x)
            features_maps[name] = x
        return features_maps, x

    @property
    def last_layer_out_channels(self):
        return list(self.conv_blocks.values())[-1][-1].conv2.out_channels 