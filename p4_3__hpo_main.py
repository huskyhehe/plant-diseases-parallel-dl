# Hyperparameter tuning

# Library imports
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, SGD
import optuna

from torchsummary import summary
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Local files imports
from hardware_info import display_cpu_info, display_gpu_info
from plant_model import PlantResNet18, PlantTrainer
from plant_constants import mean, std, num_classes, input_shape, batch_size, train_dir, valid_dir
from plot_evaluation import plot_loss_and_acc, plot_time_and_memory_usage

# Main logic
def objective(trial, num_nodes, num_workers, num_epochs, n_jobs):
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
    model.to(device)
    mode = "hpo"
    
    
    # Step 4: Model Training--------------------------------------------------------------
    # Hyperparameters
    lr_rate = trial.suggest_float('lr_rate', 1e-5, 1e-1, log=True)
    step_size = trial.suggest_int('step_size', 1, 10)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = SGD(model.fc.parameters(), lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler =  lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Create trainer
    trainer_name = f"hpo_{n_jobs}__trainer"
    trainer = PlantTrainer(trainer_name, device, model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)
    
    # Start training
    trainer.train_model()

    # Print trainer info
    trainer.display_info()
    
    return -trainer.history["valid_acc"][-1]


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: 5)')
    args = parser.parse_args()

    # Pass the objective function with the additional parameters
    study = optuna.create_study(direction='maximize') 
    study.optimize(lambda trial: objective(trial, args.num_nodes, args.num_workers, args.num_epochs), n_trials=20, n_jobs=n_jobs, callbacks=[print_best_callback])

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Store best hyper info
    with open(f"res/best_params_{n_jobs}.pkl", "wb") as file:
        pickle.dump(best_params, file)


if __name__ == "__main__":
    main()