from logging import log
import horovod
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from models.base_line.model import BaseLine
from dataloading.coco.datasets import CocoDatasets
from dataloading.coco.dataloaders import CocoDataLoaders

import torch
import pytorch_lightning as pl

import wandb

import argparse


EPOCHS = 50
NB_GPUS = -1 #all GPUs


def dataloaders(data_dir):
    coco_datasets = CocoDatasets(anns_dir=data_dir)
    coco_dataloaders = CocoDataLoaders(coco_datasets)
    props_pairs = dict(
        prop_train=.7, 
        prop_val=.3, 
        prop_test=.1
    )

    props_singles = dict(
        prop_train=1
    )

    return dict(
        pairs = coco_dataloaders.pairs(*props_pairs, batch_size=32, shuffle=True),
        singles = coco_dataloaders.singles(*props_singles, batch_size=32, shuffle=True)
    )

data = None

wandb.login()
wandb_logger = pl.loggers.WandbLogger()

def objective(trial: optuna.trial.Trial) -> float:
    
    # We optimize the number of layers, hidden units in each layer and dropouts.
    hyperparameters = dict(
        ae_lr = trial.suggest_float("ae_lr", 1e-4, 5e-4, step=1e-4),
        weight_decay_ae = trial.suggest_categorical("ae_wd", [0.1, 0.01, 0.001]),
        gan_lr = trial.suggest_float("ae_lr", 1e-4, 5e-4, step=1e-4),
        weight_decay_gan = trial.suggest_categorical("ae_wd", [0.1, 0.01, 0.001]),
    )


    model = BaseLine(
        ae_output_size=(128, 128),
        ae_bridge_out_dims=1024,
        gan_noise_dim=256,
        ae_train_dl=data["pairs"]["train"],
        disc_train_dl=data["singles"]["train"], 
        val_dl=data["pairs"]["val"], 
        test_dl=data["pairs"]["test"],
        *hyperparameters
    ) 

    trainer = pl.Trainer(
        accelerator='horovod',
        logger=wandb_logger,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        gpus=NB_GPUS if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss/acc")], # changer la loss
    )
    
    
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model) 

    return trainer.callback_metrics["val_loss/acc"].item()



if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument(
        "--data",
        "-d",
        help="Data directory"
    )

    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    data = dataloaders(os.path.abspath(args.data))
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))