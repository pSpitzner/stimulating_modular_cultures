#!/bin/bash

# example for SGE cluster
# submit from project directory !!!

#$ -S /bin/bash   # which shell to use
#$ -N brian       # name of the job
#$ -q zal.q       # which queue to use
#$ -l h_vmem=15G  # job is killed if exceeding this
#$ -cwd           # workers cd into current working directory first
#$ -o ./log/      # location for log files. directory must exist
#$ -j y

# set up environment and prevent multithreading
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# debugging variables in the log
date
uname -n

# load settings from home, activate conda environment
source /home/pspitzner/.bashrc
conda activate brian

# each worker pulls the right line with parameters from our tsv
vargs=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters.tsv)
echo "${vargs[$id]}"
${vargs[$id]}

date
