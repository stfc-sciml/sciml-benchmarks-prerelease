#!/bin/bash
sudo docker build -t samueljackson/sciml_bench .
sudo docker push samueljackson/sciml_bench
singularity build sciml_bench.img docker://samueljackson/sciml_bench:latest
