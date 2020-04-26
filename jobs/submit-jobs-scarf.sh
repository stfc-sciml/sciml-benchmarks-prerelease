#!/bin/bash

BASE_DIR=results/scarf

# Single Node
RESULTS_FOLDER=$BASE_DIR/n1-gpu1 sbatch -n 1 -N 1 --gpus-per-node=1 benchmark-scarf.sbatch
RESULTS_FOLDER=$BASE_DIR/n1-gpu2 sbatch -n 2 -N 1 --gpus-per-node=2 benchmark-scarf.sbatch
RESULTS_FOLDER=$BASE_DIR/n1-gpu4 sbatch -n 4 -N 1 --gpus-per-node=4 benchmark-scarf.sbatch

# Multi Node
RESULTS_FOLDER=$BASE_DIR/n2-gpu1 sbatch -n 2 -N 2 --gpus-per-node=1 benchmark-scarf.sbatch
RESULTS_FOLDER=$BASE_DIR/n2-gpu2 sbatch -n 4 -N 2 --gpus-per-node=2 benchmark-scarf.sbatch
RESULTS_FOLDER=$BASE_DIR/n2-gpu4 sbatch -n 8 -N 2 --gpus-per-node=4 benchmark-scarf.sbatch

RESULTS_FOLDER=$BASE_DIR/n4-gpu1 sbatch -n 4 -N 4 --gpus-per-node=1 benchmark-scarf.sbatch
RESULTS_FOLDER=$BASE_DIR/n4-gpu2 sbatch -n 8 -N 4 --gpus-per-node=2 benchmark-scarf.sbatch
RESULTS_FOLDER=$BASE_DIR/n4-gpu4 sbatch -n 16 -N 4 --gpus-per-node=4 benchmark-scarf.sbatch
