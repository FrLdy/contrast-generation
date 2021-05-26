import torch.nn as nn

from .std_blocks import UpBlock

class UnetDecoder(nn.Module):
    def __init__(self, 
        decoder_channels,
        upsampling_method,
        out_channels=3
    ):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.upsampling_method = upsampling_method
        self.up_blocks = self.build_up_blocks()
        self.last_up = UpBlock(
                *[decoder_channels[-1][1]]*2,
                decoder_channels[-1][1]+3,
                decoder_channels[-1][1],
                upsampling_method="bilinear"
        )
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
                    i, o, o*2, o,
                    self.upsampling_method
                )
            )
        return nn.ModuleDict(up_blocks)

    def forward(self, x, encoder_features_maps):
        for block, fm in zip(self.up_blocks.values(), encoder_features_maps) :
            x = block(x, fm)
        x = self.last_up(x, encoder_features_maps[-1])
        x = self.last_layer(x)
        return x