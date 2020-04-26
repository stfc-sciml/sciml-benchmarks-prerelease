#!/bin/sh
#SBATCH -p pearl
#SBATCH -t 96:00:00

SCIML_BENCH_TRACKING_URI='http://dev05.pearl.scd.stfc.ac.uk:5000'


mpirun                              \
  -x SCIML_BENCH_TRACKING_URI       \
  -x LD_LIBRARY_PATH                \
  -x PATH                           \
  -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^virbr0,lo   \
  singularity run --nv sciml-bench-tf.sif sciml-bench --using-mpi all --model-dir $RESULTS_FOLDER
