# Model Training with Multi GPUs
# DataParallel

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
from plant_model import PlantResNet18, PlantTrainer
from plant_constants import mean, std, num_classes, input_shape, batch_size, train_dir, valid_dir
from plant_hps import lr_rate, step_size, weight_decay
from plot_evaluation import plot_loss_and_acc, plot_time_and_memory_usage

# Main logic
def run(num_epochs):
    # Step 1: Check Hardware Information--------------------------------------------------
    # Check gpu
    print("CUDA available" if torch.cuda.is_available() else "CUDA unavailable")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
        
    # Step 2: Data Loading----------------------------------------------------------------
    # Define data transforms for training, validation, and testing
    
    def data_loaders(batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
        # Create datasets using ImageFolder
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        
        return train_loader, valid_loader
    
    train_loader, valid_loader = data_loaders(batch_size)
    
    
    # Step 3: Define Model----------------------------------------------------------------
    model = PlantResNet18(num_classes, False)
    model = nn.DataParallel(model)
   
    # Enable DataParallel
    print(f"Running DataParrallel on {num_gpus} GPU(s)------------------------------------")
    model.to(device)
    
    
    # Step 4: Model Training--------------------------------------------------------------
    # Hyperparameters please refer the plant_hps.py
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer for DataParralel
    optimizer = SGD(model.module.fc.parameters(), lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler =  lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Create trainer
    trainer_name = f"dp_{num_gpus}__trainer"
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
    parser = argparse.ArgumentParser(description='Run with multi-gpus -- data parallel')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: 5)')
    args = parser.parse_args()

    run(args.num_epochs)


if __name__ == "__main__":
    main()