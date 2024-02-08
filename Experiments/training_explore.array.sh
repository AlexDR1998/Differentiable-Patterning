#! /bin/sh
#$ -N channel_sampling
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=64G
export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
bash training_explore.sh $SGE_TASK_ID