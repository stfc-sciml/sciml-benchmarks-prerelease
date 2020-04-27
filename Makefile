# Makefile to create artifacts for sciml-bench

DOCKER_REMOTE=samueljackson/sciml-bench

.PHONY: sciml-bench-nvidia.sif
sciml-bench-nvidia.sif: Dockerfile.nvidia sciml-bench-nvidia.cfg
	sudo docker build -t $(DOCKER_REMOTE)-nvidia -f Dockerfile.nvidia .
	sudo docker push $(DOCKER_REMOTE)-nvidia
	sudo singularity build -F $@ sciml-bench-nvidia.cfg

.PHONY: sciml-bench-tf.sif
sciml-bench-tf.sif: 
	sudo docker build -t $(DOCKER_REMOTE)-tf -f Dockerfile.tf .
	sudo docker push $(DOCKER_REMOTE)-tf 
	sudo singularity build -F $@ sciml-bench-tf.cfg

.PHONY: all
all: sciml-bench-tf.sif sciml-bench-nvidia.sif
