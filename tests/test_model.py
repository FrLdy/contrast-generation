from unittest import TestCase

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

import pytorch_lightning as pl

import os, sys
sys.path.append(os.path.abspath("../"))


class ToyDataset(torch_data.Dataset):
    def __init__(self, len=64):
        super().__init__()
        self.len = len

    def __getitem__(self, idx):
        return torch.ones(3, 128, 128), torch.ones(3, 128, 128) * 2, torch.zeros(1)

    def __len__(self):
        return self.len

gen_train_dl    = torch_data.DataLoader(torch.ones(8, 3, 128, 128), batch_size=4)
disc_train_dl   = torch_data.DataLoader(torch.ones(8, 3, 64, 64), batch_size=4)
ae_train_dl     = torch_data.DataLoader(ToyDataset(8), batch_size=4)

class TestBaseLine(TestCase):
    def __init__(self):
        super().__init__()
        net = BaseLine(
            ae_out_channels=512,
            gan_input_channels=512 * 2 * 2,
            gan_noise_dim=512,
            ae_train_dl=ae_train_dl,
            disc_train_dl=disc_train_dl,
            gen_train_dl=gen_train_dl
        )

    def test_training_step(self):
        self.fail()
