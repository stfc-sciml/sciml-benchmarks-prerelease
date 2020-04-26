#!/bin/sh
#SBATCH -p gpu
#SBATCH -t 96:00:00

ml OpenMPI/4.0.0-GCC-8.2.0-2.31.1

DATA_DIR=/home/vol08/scarf688/data/benchmarks/
SCIML_BENCH_TRACKING_URI='http://dev05.pearl.scd.stfc.ac.uk:5000'

mpirun \
    singularity run -B $DATA_DIR:/data --nv sciml-bench-tf.sif \
        sciml-bench --tracking-uri=$SCIML_BENCH_TRACKING_URI all --data-dir /data --model-dir $RESULTS_FOLDER
