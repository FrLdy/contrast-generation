import torch
import torch.nn as nn
from .encoder import ResnetEncoder
from .bridge import Bridge
from .decoder import UnetDecoder

class ResUnetAutoencoder(nn.Module):
    def __init__(self, resnet, bridge_out_channels, upsampling_method):
        super().__init__()

        self.bridge_out_channels = bridge_out_channels

        self.encoder = ResnetEncoder(resnet)
        self.maxpool = nn.MaxPool2d(2)
        self.bridge = Bridge(
            self.encoder.last_layer_out_channels,
            bridge_out_channels
        )
        
        self.decoder = UnetDecoder(
            self.decoder_channels,
            upsampling_method
        )

    def forward(self, x):
        initial_input = x
        features_maps, x = self.encoder(x)
        x = self.maxpool(x)
        x = self.bridge(x)
        x = self.decoder(x, list(features_maps.values())[::-1]+[initial_input])
        return x         

    @property
    def decoder_channels(self):
        return (
            [(self.bridge_out_channels, self.encoder.conv_block_channel_shapes[-1][1])] 
            + [t[::-1] for t in self.encoder.conv_block_channel_shapes[::-1]]
        )

