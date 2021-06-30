import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import pytorch_lightning as pl

import itertools


from models.autoencoders.ae import ResUnetAE
from models.autoencoders.components import (
    Bridge, ConvBlock
)

from models.GANs.model import ConstrastDCGAN

class BaseLine(pl.LightningModule):
    
    resnet = torchvision.models.resnet18(pretrained=True)

    def __init__(self, bridge_out_channels, gan_noise_dim, gan_lr):
        super().__init__()
        self.ae = ResUnetAE(self.resnet, bridge_out_channels, "conv_transpose", (128, 128))

        self.conv_input_decoder = ConvBlock(bridge_out_channels*2, bridge_out_channels)

        self.bridge_union = nn.Sequential(
            nn.MaxPool2d(2),
            Bridge(self.ae.bridge.in_channels, bridge_out_channels)
        )
        
        self.gan = ConstrastDCGAN(
            latent_dim=bridge_out_channels+gan_noise_dim
        )
    
    def _ae_forward(self, batch):
        x1, x2, labels = batch
        fm1, z1 = self.encoder(x1)
        fm2, z2 = self.encoder(x2)

        z1_2 = self.ae._bridge_forward(self.sub(z1, z2)) # embd of z1 / z2
        z2_1 = self.ae._bridge_forward(self.sub(z2, z1))        
        z1u2 = self.bridge_union(self.union(z1, z2))

        return {
            "features" : (fm1, fm2), 
            "differences" : (z1_2, z2_1),
            "union" : (z1u2)
        }

    def _ae_training_step(self, batch):
        def __decoder_forward(zi, zj, fm):
            z = torch.cat((zi, zj), 1)
            z = self.conv_input_decoder(z)
            x_hat = self.decoder(z, list(fm.values())[::-1])
            return x_hat
        
        ae_res = self._ae_forward(batch)
        x1_hat, x2_hat = [
            __decoder_forward(z, ae_res["union"], fm) 
            for z, fm in (zip(ae_res["differences"], ae_res["features"]))
        ]

        self.gan.contrast_batch = ae_res["differences"] # set input for generator
        
        x1, x2, labels = batch
        losses = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        return losses

    def _gan_training_step(self, batch, optimizer_idx):
        return self.gan.training_step(batch, optimizer_idx-1)


    def training_step(self, batch, optimizer_idx):
        batch_ae, batch_gan = batch
        if optimizer_idx == 0:
            res = self._ae_training_step(batch_ae)
        else :
            res = self._gan_training_step(batch_gan, optimizer_idx)
        return res

    def _configure_ae_optimizers(self):
        params = [
            self.bridge.parameters(), 
            self.bridge_union.parameters(),
            self.conv_input_decoder.parameters(),
            self.decoder.parameters()
        ]
        return torch.optim.Adam(itertools.chain(*params), lr=self.lr)

    def _configure_gan_optimizers(self):
        return self.gan.configure_optimizers()[0]

    def configure_optimizers(self):
        return self._configure_ae_optimizers(), *self.gan.configure_optimizers()



    @property
    def decoder(self):
        return self.ae.decoder

    @property
    def encoder(self):
        return self.ae.encoder

    @property
    def bridge(self):
        return self.ae.bridge

    def union(self, zi, zj):
        return zi+zj
    
    def sub(self, zi, zj, mode="sub"):
        opes = {
            "sub" : {"fn" : torch.sub, "parameters":[zi, zj]},
            "concat" : {"fn" : torch.cat, "parameters":[(zi, zj), 1]}
        }

        return opes[mode]["fn"](*opes[mode]["parameters"])

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        ae = None
        gan = None
        return [ae, gan]

