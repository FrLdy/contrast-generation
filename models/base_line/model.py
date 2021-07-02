from logging import currentframe
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as torch_data

import pytorch_lightning as pl

import itertools


from models.autoencoders.ae import ResUnetAE
from models.autoencoders.components import (
    Bridge, ConvBlock
)

from models.GANs.model import ConstrastDCGAN

class BaseLine(pl.LightningModule):
    
    resnet = torchvision.models.resnet18(pretrained=True)

    def __init__(self, bridge_out_channels, gan_noise_dim, ae_train_dl, gen_train_dl, disc_train_dl):
        super().__init__()
        self.ae = ResUnetAE(self.resnet, bridge_out_channels, "conv_transpose", (128, 128))

        self.conv_input_decoder = ConvBlock(bridge_out_channels*2, bridge_out_channels)

        self.bridge_union = nn.Sequential(
            nn.MaxPool2d(2),
            Bridge(self.ae.bridge.in_channels, bridge_out_channels)
        )
        
        self.gan = ConstrastDCGAN(
            latent_dim=bridge_out_channels+gan_noise_dim,
            noise_dim=gan_noise_dim
        )

        self.ae_train_dl = ae_train_dl
        self.gen_train_dl = gen_train_dl
        self.disc_train_dl = disc_train_dl

        self.lr = 0.0004
    
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
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        
        return {"loss" : loss, "z_union": ae_res["union"]}

    def _gan_training_step(self, batch, optimizer_idx):
        return self.gan.training_step(batch, optimizer_idx-1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        res = None
        if optimizer_idx == 0:
            print("Train AE")
            res = self._ae_training_step(batch["ae"])
        elif optimizer_idx == 1 :
            print("Train DISC")
            res = self.gan._disc_step(batch["gan_real"], batch["gan_fake"])
        elif optimizer_idx == 2 :
            print("Train GEN")
            res = self.gan._gen_step(batch["gan_fake"])
        
        return res

    def training_epoch_end(self, outputs):
        if self.current_epoch % 2 == 0:
            self.gen_train_dl = torch_data.DataLoader([o["z_union"] for o in outputs], batch_size=self.ae_train_dl.batch_size)

    def train_dataloader(self):
        return {
            "ae"    : self.ae_train_dl, 
            "gan_real"  : self.disc_train_dl, 
            "gan_fake"   : self.gen_train_dl
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # update AE during entire epoch (1 ep on 2)
        if epoch % 2 == 0 :
            if optimizer_idx == 0 :
                optimizer.step(closure=optimizer_closure)
        # update GAN during entire epoch (1 ep on 2)
        else :
            # update discriminator 1 batch on 2
            if batch_idx % 2 == 0:
                if optimizer_idx == 1:
                    optimizer.step(closure=optimizer_closure)
            # update generator
            else : 
                if optimizer_idx == 2:
                    optimizer.step(closure=optimizer_closure)
    
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
        return [self._configure_ae_optimizers()]+self._configure_gan_optimizers(), []

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


