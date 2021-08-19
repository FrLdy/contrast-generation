from logging import log
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import os, sys
sys.path.insert(
    0,
    os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-4])
)

import contrast_generation.models as models
from contrast_generation.dataloading.coco.datasets import CocoDatasets
from contrast_generation.dataloading.coco.dataloaders import CocoDataLoaders

import torch
import pytorch_lightning as pl

import wandb

import argparse


EPOCHS = 50
NB_GPUS = -1 #all GPUs


def dataloaders(imgs_dir, anns_dir):
    coco_datasets = CocoDatasets(imgs_dir, anns_dir)
    coco_dataloaders = CocoDataLoaders(coco_datasets, 4, "sport")
    props_pairs = dict(
        prop_train=.6, 
        prop_val=.3, 
        prop_test=.1
    )

    props_singles = dict(
        prop_train=1
    )

    return dict(
        pairs = coco_dataloaders.pairs(**props_pairs, batch_size=32, shuffle=True),
        singles = coco_dataloaders.singles(**props_singles, batch_size=32, shuffle=True)
    )

data = None

wandb.login()

def objective(trial: optuna.trial.Trial) -> float:
    
    # We optimize the number of layers, hidden units in each layer and dropouts.
    hyperparameters = dict(
        ae_lr = trial.suggest_float("ae_lr", 1e-4, 5e-4, step=1e-4),
        weight_decay_ae = trial.suggest_categorical("ae_wd", [0.1, 0.01, 0.001]),
        gan_lr = trial.suggest_float("ae_lr", 1e-4, 5e-4, step=1e-4),
        weight_decay_gan = trial.suggest_categorical("ae_wd", [0.1, 0.01, 0.001]),
    )


    model = models.BaseLine(
        ae_output_size=(128, 128),
        ae_bridge_out_dims=[1024],
        gan_noise_dim=256,
        ae_train_dl=data["pairs"]["train"],
        disc_train_dl=data["singles"]["train"], 
        val_dl=data["pairs"]["val"], 
        test_dl=data["pairs"]["test"],
        **hyperparameters
    ) 

    wandb_logger = pl.loggers.WandbLogger(project="contrast_generation")

    trainer = pl.Trainer(
	gpus=4,
	num_nodes=2,
        accelerator='horovod',
        logger=wandb_logger,
	log_every_n_steps=50,
        checkpoint_callback=False,
        callbacks=[
		PyTorchLightningPruningCallback(trial, monitor="val_loss/acc"),
		pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss/acc")
	]
    )
    
    
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model) 
    trainer.test(ckpt_path=None)

    return trainer.callback_metrics["val_loss/acc"].item()



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument(
        "--imgs",
        help="Images directory"
    )
    parser.add_argument(
        "--anns",
        help="Annotations directory"
    )


    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    data = dataloaders(
        *[os.path.abspath(path) for path in (args.imgs, args.anns)]
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
