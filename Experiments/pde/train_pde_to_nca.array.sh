#! /bin/sh
#$ -N image_morph_pde
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 1 -l h_vmem=120G



bash train_pde_to_nca.sh $SGE_TASK_ID