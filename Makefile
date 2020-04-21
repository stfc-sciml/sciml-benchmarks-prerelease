# Makefile to create artifacts for sciml-bench

DOCKER_REMOTE=samueljackson/sciml-bench

sciml-bench-nvidia.sif: Dockerfile.nvidia sciml-bench-nvidia.cfg
	sudo docker build -t $(DOCKER_REMOTE)-nvidia -f Dockerfile.nvidia .
	sudo docker push $(DOCKER_REMOTE)-nvidia
	sudo singularity build $@ sciml-bench-nvidia.cfg

sciml-bench-tf.sif: Dockerfile.tf sciml-bench-tf.cfg
	sudo docker build -t $(DOCKER_REMOTE)-tf -f Dockerfile.tf .
	sudo docker push $(DOCKER_REMOTE)-tf 
	sudo singularity build $@ sciml-bench-tf.cfg

.PHONY: all
all: sciml-bench-tf.sif sciml-bench-nvidia.sif
