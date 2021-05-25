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

        self.bridge = Bridge(
            self.encoder.last_layer_out_channels,
            bridge_out_channels
        )
        
        self.decoder = UnetDecoder(
            self.decoder_channels,
            upsampling_method
        )

    def forward(self, x):
        features_maps, x = self.encoder(x)
        print(len(features_maps), features_maps.keys())
        x = nn.MaxPool2d(2)(x)
        x = self.bridge(x)
        x = self.decoder(x, list(features_maps.values())[::-1])
        return x         

    @property
    def decoder_channels(self):
        return (
            [(self.bridge_out_channels, self.encoder.conv_block_channel_shapes[-1][1])] 
            + [t[::-1] for t in self.encoder.conv_block_channel_shapes[::-1]]
        )
    

if __name__ == '__main__':
    from torchvision import models
    from std_blocks import UpBlock

    resnet = models.resnet18(pretrained=True)

    bridge_out_channels = 1024
    
    autoencoder = ResUnetAutoencoder(
        resnet,
        bridge_out_channels,
        UpBlock.UPSAMPLING_CONV_TRANSPOSE
    )

    x = torch.rand(1, 3, 572, 572)
    autoencoder(x)
    print("super")

