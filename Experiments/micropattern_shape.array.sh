#! /bin/sh
#$ -N mp_shape
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=80G

bash micropattern_shape.sh $SGE_TASK_ID