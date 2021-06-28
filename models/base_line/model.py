import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.models.gans import DCGAN

from models.autoencoders.ae import ResUnetAE
from models.autoencoders.components import Bridge

class BaseLine(pl.LightningModule):
    def __init__(self, ae:ResUnetAE, gan:DCGAN) -> None:
        super().__init__()

        self.ae = ae
        self.gan = gan

    def _ae_training_step(self, batch, batch_idx):
        x1, x2, labels = batch
        
        z1, fm1 = self.ae._compute_embedding_step(x1)
        z2, fm2 = self.ae._compute_embedding_step(x2)

        z1_c = z1 - z2
        z2_c = z2 - z1 

        x1_hat = self.ae._decoding_embedding_step((z1 - z1_c) + z1_c, fm1)
        x2_hat = self.ae._decoding_embedding_step((z2 - z2_c) + z2_c, fm2)
          
        losses = [F.mse_loss(x1_hat, x1), F.mse_loss(x2_hat, x2)]
        return losses
        



        

