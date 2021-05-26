import torch.nn as nn
from collections import OrderedDict


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

    @property
    def last_layer_out_channels(self):
        return list(self.conv_blocks.values())[-1][-1].conv2.out_channels 