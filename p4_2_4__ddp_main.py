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


from torchsummary import summary
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Local files imports
from hardware_info import display_cpu_info, display_gpu_info
from plant_model import PlantResNet18, PlantTrainer
from plant_constants import mean, std, num_classes, input_shape, batch_size, train_dir, valid_dir
from plant_hps import lr_rate, step_size, weight_decay
from plot_evaluation import plot_loss_and_acc, plot_time_and_memory_usage


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
    
# Main logic
def run(rank: int, world_size: int, num_nodes: int, num_workers: int, num_epochs: int):
    # Step 1: Check Hardware Information-------------------------------------------------------
    display_cpu_info()
    # Check gpu
    print("CUDA available" if torch.cuda.is_available() else "CUDA unavailable")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Step 2: Data Loading---------------------------------------------------------------------
    torch.manual_seed(0)
    # Define data transforms for training, validation, and testing
    def data_loaders(num_workers: int, batch_size):
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        
        return train_loader, valid_loader
    
    train_loader, valid_loader = data_loaders(num_workers, batch_size)
    
    
    # Step 3: Define Model------------------------------------------------------------------------
    setup(rank, world_size)
    
    # create model and move it to GPU with id rank
    model = PlantResNet18(num_classes).to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])

    print(f"Running on rank {rank}---------------------------------")
    torch.cuda.set_device(rank)
    mode = "ddp"

    
    # Step 4: Model Training----------------------------------------------------------------------
    # Hyperparameters please refer the plant_hps.py
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = SGD(ddp_model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler =  lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Create trainer
    trainer_name = f"ddp_{rank}__trainer"
    trainer = PlantTrainer(trainer_name, rank, ddp_model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)
    
    # Start training
    trainer.train_model()

    # Print trainer info
    trainer.display_info()

    # Store trainer info
    with open(f"res/{trainer_name}.pkl", "wb") as file:
        pickle.dump(trainer, file)
    
    
    # Step 5: Model Evaluation---------------------------------------------------------------------
    epoch_list = [i + 1 for i in range(num_epochs)]
    
    plot_loss_and_acc(mode, num_nodes, epoch_list, trainer.history)
    plot_time_and_memory_usage(mode, num_nodes, epoch_list, trainer.history)

    cleanup()


def main(run_func, world_size):
    parser = argparse.ArgumentParser(description='Run with multi-gpus -- distributed data parallel')
    parser.add_argument('--world_size', type=int, default=world_size, help='total number of processes')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: 5)')
    args = parser.parse_args()

    mp.spawn(run_func, args=(args.world_size, args.num_nodes, args.num_workers, args.num_epochs), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Running DistributedDataParralel on {num_gpus} GPUs----------------------")
    main(run, num_gpus)