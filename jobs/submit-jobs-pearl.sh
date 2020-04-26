#!/bin/bash

BASE_DIR=results/pearl

# Single Node
RESULTS_FOLDER=$BASE_DIR/n1-gpu1 sbatch -n 1 --gres=gpu:1 -w mn2 benchmark-pearl.sbatch
RESULTS_FOLDER=$BASE_DIR/n1-gpu2 sbatch -n 2 --gres=gpu:2 -w mn2 benchmark-pearl.sbatch
RESULTS_FOLDER=$BASE_DIR/n1-gpu4 sbatch -n 4 --gres=gpu:4 -w mn2 benchmark-pearl.sbatch
RESULTS_FOLDER=$BASE_DIR/n1-gpu8 sbatch -n 8 --gres=gpu:8 -w mn2 benchmark-pearl.sbatch
RESULTS_FOLDER=$BASE_DIR/n1-gpu16 sbatch -n 16 --gres=gpu:16 -w mn2 benchmark-pearl.sbatch

# Multi Node
RESULTS_FOLDER=$BASE_DIR/n2-gpu32 sbatch -n 32 -N 2 --gpus-per-node=16 benchmark-pearl.sbatch
