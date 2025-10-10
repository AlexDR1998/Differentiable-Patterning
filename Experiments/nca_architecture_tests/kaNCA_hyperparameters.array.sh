#! /bin/sh
#$ -N image_kaNCA_tune
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 1 -l h_vmem=120G



bash kaNCA_hyperparameters.sh $SGE_TASK_ID