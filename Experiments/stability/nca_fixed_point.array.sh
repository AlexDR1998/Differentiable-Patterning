#! /bin/sh
#$ -N image_morph_fixed_point
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l rl9=true

#$ -q gpu -l gpu=1 -pe sharedmem 1 -l h_vmem=32G



bash nca_fixed_point.sh $SGE_TASK_ID