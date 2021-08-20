#!/bin/bash

# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

#SBATCH --exclusive
#SBATCH --time 48:00:00
#SBATCH --mem 10000 
#SBATCH --mail-type ALL
#SBATCH --mail-user francois.ledoyen@unicaen.fr
#SBATCH --partition gpu_all
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --output %J.out
#SBATCH --error %J.err
#SBATCH --cpus-per-task=10
#SBATCH --tasks-per-node=2

# Loading the required modules
module load python3-DL/3.8.5


logs_dir="$SLURM_SUBMIT_DIR/logs"
mkdir -p $logs_dir
mkdir "$logs_dir/$SLURM_JOB_ID"
mv $SLURM_JOB_ID.* "$logs_dir/$SLURM_JOB_ID/"


BASE_DIR="/home/2017025/fledoy01/code/contrast_generation/data/coco"
IMGS_DIR="$BASE_DIR/imgs"
ANNS_DIR="$BASE_DIR/datasets"
srun horovodrun -np 4 python train.py -p --imgs $IMGS_DIR --anns $ANNS_DIR 
