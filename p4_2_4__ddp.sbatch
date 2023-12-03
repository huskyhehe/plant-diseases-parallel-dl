#!/bin/bash
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=4                 
#SBATCH --mem=16G          
#SBATCH --time=8:00:00            
#SBATCH --gres=gpu:p100:4

# module load anaconda3/2022.05 cuda/11.7

# conda create --name pytorch_env python=3.9 -y
source activate pytorch_env
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y


python p4_2_4__ddp_main.py --num_nodes 1 --num_workers 4 --num_epochs 5

