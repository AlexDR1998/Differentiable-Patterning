#! /bin/sh
#$ -N nPDE_train
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=120G


. /etc/profile.d/modules.sh

export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
module load anaconda
source activate jax_gpu

python ./train_pde_to_nca.py 
source deactivate