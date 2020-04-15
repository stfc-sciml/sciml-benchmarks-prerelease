#!/bin/bash
sudo docker build -t samueljackson/sciml_bench .
sudo docker push samueljackson/sciml_bench
sudo singularity build sciml_bench.sif sciml_bench.cfg
sudo singularity push -U sciml_bench.sif library://sljack/default/sciml-bench
