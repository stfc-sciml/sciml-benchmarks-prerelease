# Makefile to create artifacts for sciml-bench

DOCKER_REMOTE=samueljackson/sciml-bench

.PHONY: sciml-bench-nvidia.sif
sciml-bench-nvidia.sif: sciml-bench-nvidia.cfg
	sudo SINGULARITY_TMPDIR=/home/lhs18285 singularity build -F $@ sciml-bench-nvidia.cfg


openmpi.sif: openmpi.cfg
	sudo SINGULARITY_TMPDIR=/home/lhs18285 singularity build -F $@ openmpi.cfg


sciml-bench-tf.sif: openmpi.sif sciml-bench-tf.cfg
	sudo singularity build -F $@ sciml-bench-tf.cfg

.PHONY: all
all: sciml-bench-tf.sif sciml-bench-nvidia.sif
