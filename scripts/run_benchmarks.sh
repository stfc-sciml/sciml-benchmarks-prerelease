#!/bin/bash

RESULTS_FOLDER=results/gpu1 sbatch --gres=gpu:1 -w mn2 benchmark.bsub
RESULTS_FOLDER=results/gpu2 sbatch --gres=gpu:2 -w mn2 benchmark.bsub
RESULTS_FOLDER=results/gpu4 sbatch --gres=gpu:4 -w mn2 benchmark.bsub
RESULTS_FOLDER=results/gpu8 sbatch --gres=gpu:8 -w mn2 benchmark.bsub
RESULTS_FOLDER=results/gpu16 sbatch --gres=gpu:16 -w mn2 benchmark.bsub
