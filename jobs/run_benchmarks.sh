#!/bin/sh

# -bind-to none                     \
# -map-by slot                      \
# -mca btl openib,self -mca pml ob1 \


mpirun -np 2                        \
  -H cn2g24:1,cn2g26:1              \
  -x SCIML_BENCH_TRACKING_URI=''    \
  -x LD_LIBRARY_PATH     -x PATH     -x HOROVOD_MPI_THREADS_DISABLE=1     -x NCCL_SOCKET_IFNAME=^virbr0,lo   \
  singularity run --nv -B ~/data/:/data sciml-bench-tf.sif sciml-bench --using-mpi em-denoise /data/benchmarks/em_denoise out
