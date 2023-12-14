#!/bin/bash
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4                         
#SBATCH --time=8:00:00      

# Interactive commands:
# srun -p courses-gpu --gres=gpu:4 --cpus-per-task=4 --time=8:00:00 --pty /bin/bash


module load anaconda3/2022.05 cuda/11.7
source activate new_env

python p4_2_3__ddp_main.py --num_epochs 5

