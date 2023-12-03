#!/bin/bash
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4                 
#SBATCH --mem=16G          
#SBATCH --time=8:00:00      
#SBATCH --gres=gpu:p100:4


# Sometimes we cannot get expected gpu, so we use interactive  commands most of time:
# srun -p courses-gpu --nodes=1 --gres=gpu:p100:4 --cpus-per-task=4 --pty /bin/bash


module load anaconda3/2022.05 cuda/11.7
# conda create --name new_env python=3.11 -y
source activate new_env
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
# conda install torchsummary

python p4_2_3__dp_main.py --num_nodes 1 --num_workers 4 --num_epochs 5

