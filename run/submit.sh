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

source /home/pspitzner/.bashrc
conda activate brian

# vargs=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters_topo.tsv)
# echo "${vargs[$id]}"
# ${vargs[$id]}

# vargs=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters_dyn.tsv)
# echo "${vargs[$id]}"
# ${vargs[$id]}

vargs=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters_stim_02.tsv)
echo "${vargs[$id]}"
${vargs[$id]}

vargs=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters_stim_012.tsv)
echo "${vargs[$id]}"
${vargs[$id]}

vargs=$(awk "NR==$(($SGE_TASK_ID + 1))" ./run/parameters_stim_0123.tsv)
echo "${vargs[$id]}"
${vargs[$id]}

date
