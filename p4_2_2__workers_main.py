# Model Training with Single GPU
# Multi-worker

# Library imports
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, SGD

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

# Main logic
def run(num_workers, num_epochs):
    # Step 1: Check Hardware Information--------------------------------------------------
    display_cpu_info()
    # Check gpu
    print("CUDA available" if torch.cuda.is_available() else "CUDA unavailable")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
        
    # Step 2: Data Loading----------------------------------------------------------------
    torch.manual_seed(0)
    # Define data transforms for training, validation, and testing
    def data_loaders(num_workers, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
        # Create datasets using ImageFolder
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)
        
        return train_loader, valid_loader
    
    train_loader, valid_loader = data_loaders(num_workers, batch_size)
    
    
    # Step 3: Define Model----------------------------------------------------------------
    model = PlantResNet18(num_classes, False)
    
    print(f"Running on {num_gpus} GPU(s) with {num_workers} worker(s)---------------------")
    model.to(device)
    
    
    # Step 4: Model Training--------------------------------------------------------------
    # Hyperparameters please refer the plant_hps.py
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = SGD(model.fc.parameters(), lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler =  lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Create trainer
    trainer_name = f"workers_{num_workers}__trainer"
    trainer = PlantTrainer(trainer_name, device, model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)
    
    # Start training
    trainer.train_model()

    # Print trainer info
    trainer.display_info()

    # Store trainer info
    with open(f"res/{trainer_name}.pkl", "wb") as file:
        pickle.dump(trainer, file)
    
    
    # Step 5: Model Evaluation---------------------------------------------------------------
    epoch_list = [i + 1 for i in range(num_epochs)]
    
    plot_loss_and_acc(trainer_name, epoch_list, trainer.history)
    plot_time_and_memory_usage(trainer_name, epoch_list, trainer.history)


def main():
    parser = argparse.ArgumentParser(description='Run with multi-process -- workers')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (default: 0)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: 5)')
    args = parser.parse_args()

    run(args.num_workers, args.num_epochs)


if __name__ == "__main__":
    main()