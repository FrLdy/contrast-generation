import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


from models.autoencoders.components import (
    ResnetEncoder,
    Bridge, 
    UnetDecoder
)


class ResUnetAE(pl.LightningModule):
    def __init__(self, resnet, bridge_out_channels, upsampling_method, output_size, copy_n_crop=True, lr=1e-4):
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
            upsampling_method,
            output_size=output_size,
            copy_n_crop=copy_n_crop
        )

        self.lr = lr

    def forward(self, x):
        features_maps, x = self.encoder(x)
        x = self.maxpool(x)
        x = self.bridge(x)
        return x         

    @property
    def decoder_channels(self):
        return (
            [(self.bridge_out_channels, self.encoder.conv_block_channel_shapes[-1][1])] 
            + [t[::-1] for t in self.encoder.conv_block_channel_shapes[::-1]]
        )

    def training_step(self, batch, batch_idx):
        x, labels = batch
        loss = self.shared_step(x) 
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        loss = self.shared_step(x)
        self.log('val_loss', loss, prog_bar=True )
        return loss

    def shared_step(self, batch):
        z, features_maps = self._compute_embedding_step(batch)
        x_hat = self._decoding_embedding_step(z, features_maps)
        return F.mse_loss(x_hat, batch)

    def _compute_embedding_step(self, batch):
        x = batch
        with torch.no_grad():
            features_maps, x = self.encoder(x)
        x = self.maxpool(x)
        z = self.bridge(x)
        return z, features_maps

    def _decoding_embedding_step(self, z, features_maps):
        x_hat = self.decoder(z, list(features_maps.values())[::-1])
        return x_hat
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)