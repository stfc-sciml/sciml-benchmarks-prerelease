#!/bin/sh
#SBATCH -p pearl
#SBATCH -t 96:00:00

ml OpenMPI/4.0.1-GCC-8.3.0-2.32

DATA_DIR=/mnt/beegfs/work/stfc/pearl008/data
SCIML_BENCH_TRACKING_URI='http://dev05.pearl.scd.stfc.ac.uk:5000'

mpirun \
    singularity run -B $DATA_DIR:/data --nv sciml-bench-tf.sif \
        sciml-bench --tracking-uri=$SCIML_BENCH_TRACKING_URI all --data-dir /data --model-dir $RESULTS_FOLDER
