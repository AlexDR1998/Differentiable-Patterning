#! /bin/sh
#$ -N mp_nca_averaged
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l rl9=true

#$ -q gpu -l gpu=1 -pe sharedmem 1 -l h_vmem=80G


. /etc/profile.d/modules.sh

export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
module load anaconda
source activate jax_gpu

python ./micropattern_averaged.py $1
source deactivate