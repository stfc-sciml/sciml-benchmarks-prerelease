#!/bin/bash
sudo docker build -t samueljackson/sciml_bench .
sudo docker push samueljackson/sciml_bench
singularity build sciml_bench.sif docker://samueljackson/sciml_bench:latest
