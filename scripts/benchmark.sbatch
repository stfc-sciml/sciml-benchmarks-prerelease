#!/bin/sh
#SBATCH -p pearl
#SBATCH -n 2
#SBATCH -t 96:00:00
#SBATCH --mem 400G

singularity exec \
	--nv \
	sciml-bench em_denoise ./data $RESULTS_FOLDER --config /workspace/examples/em_denoise.yml

singularity exec \
	--nv \
	sciml-bench dms_classifier ./data $RESULTS_FOLDER --config /workspace/examples/dms_classifier.yml

singularity exec \
	--nv \
	sciml-bench slstr_cloud ./data $RESULTS_FOLDER --config /workspace/examples/slstr_cloud.yml
