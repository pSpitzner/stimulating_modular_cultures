#!/bin/bash

# submit from project directory !!!

#$ -S /bin/bash
#$ -N brian
#$ -q rostam.q
#$ -l h_vmem=15G # job is killed if exceeding this
#$ -cwd
#$ -o ./log/
#$ -j y

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

date
if [ $id -eq 0 ]; then
  ssh -t sohrab001 "/home/pspitzner/bin/notify started $JOB_NAME.$TASK_ID.$JOB_ID"
fi


vargs=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters.tsv)
echo "${vargs[$id]}"

${vargs[$id]}

date
if [ $id -eq 809 ]; then
  ssh -t sohrab001 "/home/pspitzner/bin/notify finished $JOB_NAME.$TASK_ID.$JOB_ID"
fi
