#!/bin/bash

# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

#SBATCH --exclusive
#SBATCH --time 48:00:00
#SBATCH --mem 10000 
#SBATCH --mail-type ALL
#SBATCH --mail-user francois.ledoyen@unicaen.fr
#SBATCH --partition gpu_p100
#SBATCH --gres gpu:2
#SBATCH --nodes 2
#SBATCH --output %J.out
#SBATCH --error %J.err
#SBATCH --cpus-per-task=10
#SBATCH --tasks-per-node=2

# Loading the required modules
module load python3-DL/3.8.5

# Starting the calculation
srun ./train.sh > std_out.log

mkdir $SLURM_SUBMIT_DIR/logs/$SLURM_JOB_ID
mv *.log $SLURM_SUBMIT_DIR/logs/$SLURM_JOB_ID/ 
