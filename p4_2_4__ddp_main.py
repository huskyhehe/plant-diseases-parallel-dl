# Model Training with GPUs
# DistributedDataParralel

# Library imports
import os
import sys
import tempfile
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import lr_scheduler, SGD

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp


import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Local files imports
from plant_model import PlantResNet18, PlantTrainerDistributed
from plant_constants import mean, std, num_classes, input_shape, batch_size, train_dir, valid_dir
from plant_hps import lr_rate, step_size, weight_decay
from plot_evaluation import plot_loss_and_acc, plot_time_and_memory_usage


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
# Main logic
def run(rank: int, world_size: int, num_epochs: int):
    # Step 1: Check Hardware Information-------------------------------------------------------
    # Check gpu
    print("CUDA available" if torch.cuda.is_available() else "CUDA unavailable")
    # Set device
    setup(rank, world_size)
    # device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    set_seed(0)


    # Step 2: Data Loading---------------------------------------------------------------------
    # Define data transforms for training, validation, and testing
    def data_loaders(batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
        # Create datasets using ImageFolder
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)

         # Use DistributedSampler for distributed training
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)
        
        return train_loader, valid_loader
    
    train_loader, valid_loader = data_loaders(batch_size)
    
    
    # Step 3: Define Model------------------------------------------------------------------------

    
    # create model and move it to GPU with id rank
    model = PlantResNet18(num_classes, False)
    model.to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    
    print(f"Running on rank {rank}---------------------------------")

    
    # Step 4: Model Training----------------------------------------------------------------------
    # Hyperparameters please refer the plant_hps.py
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = SGD(ddp_model.parameters(), lr=lr_rate, momentum=momentum, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler =  lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Create trainer
    trainer_name = f"ddp_{rank}__trainer"
    # trainer = PlantTrainer(trainer_name, rank, ddp_model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)
    trainer = PlantTrainerDistributed(rank, world_size, trainer_name, rank, ddp_model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)

    print(f"trainer.rank:{trainer.rank}")
    print(f"trainer.world_size:{trainer.world_size}")
    print(f"trainer.device: {trainer.device}")
    print(f"trainer.criterion: {trainer.criterion}")
    print(optimizer)
    print(scheduler)
    
    # Start training
    trainer.train_model()

    # Print trainer info
    trainer.display_info()

    # Store trainer info
    with open(f"res/{trainer_name}.pkl", "wb") as file:
        pickle.dump(trainer, file)
    
    
    # Step 5: Model Evaluation---------------------------------------------------------------------
    epoch_list = [i + 1 for i in range(num_epochs)]
    
    plot_loss_and_acc(trainer_name, epoch_list, trainer.history)
    plot_time_and_memory_usage(trainer_name, epoch_list, trainer.history)

    cleanup()


def main(run_func, world_size):
    parser = argparse.ArgumentParser(description='Run with multi-gpus -- distributed data parallel')
    parser.add_argument('--world_size', type=int, default=world_size, help='total number of gpus')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: 5)')
    args = parser.parse_args()

    torch.cuda.manual_seed_all(0)
    mp.spawn(run_func, args=(args.world_size, args.num_epochs), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Running DistributedDataParralel on {num_gpus} GPUs----------------------")
    main(run, num_gpus)