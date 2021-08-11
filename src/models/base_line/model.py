import torch
import torch.functional as F

import pytorch_lightning as pl
from pl_bolts.models.gans import DCGAN

from models.autoencoders.ae import ResUnetAE

class BaseLine(pl.LightningModule):
    def __init__(self, ae:, gan) -> None:
        super().__init__()

        self.ae = ae
        self.gan = gan

    def _ae_training_step(self, batch, batch_idx):
        x1, x2, labels = batch

        self.ae.shared_step

