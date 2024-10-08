#! /bin/sh
#$ -N neural_pde_tune
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l rl9=true

#$ -q gpu -l gpu=1 -pe sharedmem 1 -l h_vmem=80G



bash micropattern_averaged.sh $SGE_TASK_ID