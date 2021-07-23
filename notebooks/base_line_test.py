# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from torchsummary import summary
from torchvision import models
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import json
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import os, sys
sys.path.append(os.path.abspath("../"))


from models.base_line.model import BaseLine
from models.autoencoders.components import ConvBlock, Bridge
from models.GANs.model import ConstrastDCGAN


# %%
class ToyDataset(torch_data.Dataset):
    def __init__(self, len=64):
        super().__init__()
        self.len = len
    
    def __getitem__(self, idx):
        return torch.ones(3, 128, 128), torch.ones(3, 128, 128)*2, torch.zeros(1)

    def __len__(self):
        return self.len


# %%
gen_train_dl    = torch_data.DataLoader(torch.ones(8, 3, 128, 128), batch_size=4)
disc_train_dl   = torch_data.DataLoader(torch.ones(8, 3, 64, 64), batch_size=4)
ae_train_dl     = torch_data.DataLoader(ToyDataset(8), batch_size=4)

val_dl          = torch_data.DataLoader(ToyDataset(8), batch_size=4) 
test_dl         = torch_data.DataLoader(ToyDataset(8), batch_size=4)


# %%
z_size = 2048


# %%
net = BaseLine(
            ae_output_size=(128, 128),
            ae_bridge_out_dims=[512],
            gan_noise_dim=512,
            ae_train_dl=ae_train_dl,
            disc_train_dl=disc_train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            ae_lr=1e-4, gan_lr=1e-4
        )


# %%
net.forward(*net.example_input_array).shape


# %%
real_batch=next(iter(disc_train_dl))
real_batch.shape


# %%
fake_batch=net._ae_forward(*next(iter(ae_train_dl)))["union"].flatten(1)
fake_batch.shape


# %%
net.gan._disc_step(
    real=real_batch, 
    fake=fake_batch
)


# %%
net.gan._gen_step(fake_batch)


# %%
trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=2)
trainer.fit(net)


# %%
trainer.test(net)

# %% [markdown]
# logger = TensorBoardLogger("tb_logs/", name="my_model", log_graph=True)
# logger.log_graph(net, net.example_input_array)
# %load_ext tensorboard
# %tensorboard --logdir tb_logs

# %%
torch.Tensor(4)


# %%
torch.cat([torch.Tensor([0.1])]*2, 0).mean()


# %%



