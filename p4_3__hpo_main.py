# Hyperparameter tuning

# Library imports
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, SGD

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Local files imports
from hardware_info import display_cpu_info, display_gpu_info
from plant_model import PlantResNet18, PlantTrainerTuning
from plant_constants import mean, std, num_classes, input_shape, batch_size, train_dir, valid_dir
from plot_evaluation import plot_loss_and_acc, plot_time_and_memory_usage


def objective(trial, num_workers, num_epochs, n_jobs):
    print(f"Trial {trial.number} starts--------------------------------------------------")
    # Step 1: Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Step 2: Data Loading----------------------------------------------------------------
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

    # Step 4: Model Training--------------------------------------------------------------
    # Hyperparameters
    lr_rate = trial.suggest_float('lr_rate', 1e-5, 1e-1, log=True)
    step_size = trial.suggest_int('step_size', 3, 10)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.89, 0.99)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = SGD(model.fc.parameters(), lr=lr_rate, momentum=momentum, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler =  lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Create trainer
    trainer_name = f"hpo_{n_jobs}__trainer"
    trainer = PlantTrainerTuning(trainer_name, device, model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, trial=trial)
    
    # Start training
    trainer.train_model()
    return trainer.best_valid_acc


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials (default: 10)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs (default: 1)')
    args = parser.parse_args()

    # multiprocessing.set_start_method('spawn', force=True)
    storage = JournalStorage(JournalFileStorage("./journal.log"))
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), storage=storage, load_if_exists=True)
    sequential_start = time.time()
    study.optimize(lambda trial: objective(trial, 4, args.num_epochs, args.n_jobs), n_trials=args.n_trials, n_jobs=args.n_jobs, callbacks=[print_best_callback])
    elapsed_time = time.time() - sequential_start
    sequential_best_value = study.best_value
    
    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Elapsed time: {elapsed_time} s")
    
    with open(f"res/hpo_time_{args.n_jobs}.pkl", "wb") as file:
        pickle.dump(elapsed_time, file)

    # Store best hyper info
    with open(f"res/best_hps.pkl", "wb") as file:
        pickle.dump(best_params, file)


if __name__ == "__main__":
    main()