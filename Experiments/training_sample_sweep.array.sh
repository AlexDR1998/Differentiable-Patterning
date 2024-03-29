#! /bin/sh
#$ -N sample_sweep
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=80G
export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
bash training_sample_sweep.sh $SGE_TASK_ID