from logging import currentframe
import torch
from torch.autograd.grad_mode import set_grad_enabled
import torch.nn as nn
import torch.optim
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
from losses.contrast_loss import ContentContrastLoss, TestContentContrastLoss

class BaseLine(pl.LightningModule):

    
    resnet = torchvision.models.resnet18(pretrained=True)

    def __init__(self, 
                ae_output_size,
                ae_bridge_out_dims, 
                gan_noise_dim, 
                ae_train_dl, disc_train_dl,
                val_dl, test_dl,
                ae_lr, gan_lr,
                weight_decay_ae, weight_decay_gan
            ):
        super().__init__()

        self.ae_lr = ae_lr
        self.gan_lr = gan_lr
        self.weight_decay_ae = weight_decay_ae
        self.weight_decay_gan = weight_decay_gan
        

        self.ae = ResUnetAE(self.resnet, ae_bridge_out_dims, "conv_transpose", ae_output_size)
        self.bridge_union = nn.Sequential(
            nn.MaxPool2d(2),
            Bridge(self.ae.bridge.in_channels, ae_bridge_out_dims)
        )

        self.conv_input_decoder = ConvBlock(ae_bridge_out_dims[-1]*2, ae_bridge_out_dims[-1])
        
        self.gan = ConstrastDCGAN(
            latent_dim=ae_bridge_out_dims[-1]*4,
            noise_dim=gan_noise_dim,
            learning_rate=self.gan_lr,
            weight_decay=self.weight_decay_gan
        )

        self.ae_train_dl = ae_train_dl
        self.disc_train_dl = disc_train_dl

        self.test_dl = test_dl
        self.val_dl = val_dl

        try:
            self.contrast_loss = ContentContrastLoss()
        except:
            self.contrast_loss = TestContentContrastLoss()


    def forward(self, x1, x2, label):
        ae_output = self._ae_forward(x1, x2, label)["union"]
        res = self._gan_forward(ae_output)
        
        return res


    def _gan_forward(self, z):
        return self.gan.forward(z.flatten(1))
        

    def _ae_forward(self, x1, x2, label):
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


    def _ae_shared_step(self, batch):

        def __decoder_forward(zi, zj, fm):
            z = torch.cat((zi, zj), 1)
            z = self.conv_input_decoder(z)
            x_hat = self.decoder(z, list(fm.values())[::-1])
            
            return x_hat

        x1, x2, label = batch
        ae_res = self._ae_forward(x1, x2, label)
        x1_hat, x2_hat = [
            __decoder_forward(z, ae_res["union"], fm) 
            for z, fm in (zip(ae_res["differences"], ae_res["features"]))
        ]

        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        
        return {"loss" : loss, "z_union": ae_res["union"]}


    def training_step(self, batch, batch_idx, optimizer_idx):
        res = None
        if optimizer_idx == 0:
            res = self._ae_shared_step(batch["ae"])
            self.log("train_loss/ae", res["loss"])
        else:           
            batch_fake = self._ae_shared_step(batch["gan_fake"])["z_union"].flatten(1)
            if optimizer_idx == 1 :
                res = self.gan._get_disc_loss(batch["gan_real"], batch_fake)
                self.log("train_loss/disc", res)
            
            elif optimizer_idx == 2 :
                res = self.gan._get_gen_loss(batch_fake)
                self.log("train_loss/gen", res)
        
        return res

    @property
    def is_ae_epoch(self):
        return self.current_epoch % 2 == 0

    def train_dataloader(self):
        if self.is_ae_epoch:
            return {"ae" : self.ae_train_dl}
 
        return {
            "gan_real"  : self.disc_train_dl, 
            "gan_fake"   : self.ae_train_dl
        }

    def validation_step(self, batch, batch_idx):
        if self.is_ae_epoch:
            x1, x2, label = batch
            ae_val = self._ae_shared_step(batch)
            ae_loss = ae_val["loss"]
            disc_loss = self.gan._get_gen_loss(ae_val["z_union"].flatten(1))
            gen_loss = self.contrast_loss(x1, x2, self.forward(x1, x2, label))
            metrics = {
                "val_loss/ae": ae_loss,
                "val_loss/disc": disc_loss,
                "val_loss/acc":gen_loss
            }
            self.log_dict(metrics)
            return metrics
    
    def val_dataloader(self):
        return self.val_dl


    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        for k in ("ae", "gan"):
            metrics[f"test_loss/{k}"] = metrics.pop(f"val_loss/{k}")
        self.log_dict(metrics)        

    def test_dataloader(self):
        return self.test_dl

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # update AE during entire epoch
        if epoch % 2 == 0 :
            if optimizer_idx == 0 :
                optimizer.step(closure=optimizer_closure)
        # update GAN during entire epoch
        elif batch_idx % 2 == 0 and optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)
        # update generator
        elif batch_idx % 1 == 0 and optimizer_idx == 2:
            optimizer.step(closure=optimizer_closure)

    def _configure_ae_optimizers(self):
        params = [
            self.bridge.parameters(), 
            self.bridge_union.parameters(),
            self.conv_input_decoder.parameters(),
            self.decoder.parameters()
        ]
        return torch.optim.Adam(itertools.chain(*params), lr=self.ae_lr, weight_decay=self.weight_decay_ae)

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


    @property
    def example_input_array(self):
        return torch.zeros((3, 3, 128, 128)), torch.zeros((3, 3, 128, 128)), torch.ones(1)