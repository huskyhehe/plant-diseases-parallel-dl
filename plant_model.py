import torch
import torch.nn as nn
from torchvision import datasets, models
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import optuna


# This is the model class based on ResNet18
class PlantResNet18(nn.Module):
    def __init__(self, num_classes, extract_features=True):
        super(PlantResNet18, self).__init__()
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if extract_features else None)

        if extract_features:
            for param in resnet18.parameters():
                param.requires_grad = False
        
        # Remove the fully connected layer
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  
        self.fc = nn.Linear(resnet18.fc.in_features, num_classes)
        
        # Prevent overfitting only when using pre-trained weights
        self.dropout = nn.Dropout(p=0.1) if extract_features else nn.Identity()

    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        x = self.dropout(x)
        return x


# This is the trainer class that modulized model training and validation process
class PlantTrainer:
    def __init__(self, name, device, model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs):
        self.name = name
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.best_valid_acc = 0.0
        self.total_time = 0.0
        self.history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": [], "epo_elapsed_time": [], "max_alloc" : []}

    def train_model(self):
        start_time = time.time()

        # Train the model for the specified number of epochs
        for epoch in range(self.num_epochs):
            epo_start_time = time.time()
            
            # Set the model to train mode
            self.model.train()
            # Training loop
            running_loss, running_corrects = self._run_loader(self.train_loader, is_training=True)

            # Calculate the train loss and accuracy
            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects.double() / len(self.train_loader.dataset)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Set the model to evaluation mode
            self.model.eval()
            # Validation loop
            running_loss, running_corrects = self._run_loader(self.valid_loader, is_training=False)

            # Calculate the validation loss and accuracy
            valid_loss = running_loss / len(self.valid_loader.dataset)
            valid_acc = running_corrects.double() / len(self.valid_loader.dataset)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_acc"].append(valid_acc)

            epo_elapsed_time = time.time() - epo_start_time
            self.history["epo_elapsed_time"].append(epo_elapsed_time)

            max_alloc = torch.cuda.max_memory_allocated(device=self.device) /  (1024 ** 2)  # in MB
            self.history["max_alloc"].append(max_alloc)

            # Print the epoch results
            print(f"Epoch [{epoch + 1}/{self.num_epochs}]------------------------------------------------------------------------")
            print(f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc:.4f}")
            print(f"| Elapsed Time: {epo_elapsed_time:.4f} s | Max GPU Memory Alloc: {max_alloc:.4f} MB")

            # Check if the current validation accuracy is the best so far
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "history": self.history
                }
                torch.save(checkpoint, f"res/{self.name}_best.pth")

            # Step the learning rate scheduler
            self.scheduler.step()

        self.total_time = time.time() - start_time
        
    def _run_loader(self, loader, is_training):
        running_loss = 0.0
        running_corrects = 0

        with torch.set_grad_enabled(is_training):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if is_training:
                    # Zero the optimizer gradients
                    self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimizer step if in training mode
                if is_training:
                    loss.backward()
                    self.optimizer.step()

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if is_training:
              self.scheduler.step()

        return running_loss, running_corrects

    def display_info(self):
        print(f"\ntotal_time: {self.total_time}")
        print(f"best_valid_acc: {self.best_valid_acc}")
        for k, v in self.history.items():
            print(f"{k}: {v}")


class PlantTrainerDistributed(PlantTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override
    def train_model(self):
        start_time = time.time()

        # Train the model for the specified number of epochs
        for epoch in range(self.num_epochs):

            self.train_loader.sampler.set_epoch(epoch)
            self.valid_loader.sampler.set_epoch(epoch)
            
            epo_start_time = time.time()
            
            # Set the model to train mode
            self.model.train()
            # Training loop
            running_loss, running_corrects = self._run_loader(self.train_loader, is_training=True)

            # Calculate the train loss and accuracy
            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects / len(self.train_loader.dataset)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Set the model to evaluation mode
            self.model.eval()
            # Validation loop
            running_loss, running_corrects = self._run_loader(self.valid_loader, is_training=False)

            # Calculate the validation loss and accuracy
            valid_loss = running_loss / len(self.valid_loader.dataset)
            valid_acc = running_corrects / len(self.valid_loader.dataset)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_acc"].append(valid_acc)

            epo_elapsed_time = time.time() - epo_start_time
            self.history["epo_elapsed_time"].append(epo_elapsed_time)

            max_alloc = torch.cuda.max_memory_allocated(device=self.device) /  (1024 ** 2)  # in MB
            self.history["max_alloc"].append(max_alloc)

            # Print the epoch results
            print(f"Epoch [{epoch + 1}/{self.num_epochs}]------------------------------------------------------------------------")
            print(f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc:.4f}")
            print(f"| Elapsed Time: {epo_elapsed_time:.4f} s | Max GPU Memory Alloc: {max_alloc:.4f} MB")

            # Check if the current validation accuracy is the best so far
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "history": self.history
                }
                torch.save(checkpoint, f"res/{self.name}_best.pth")

            # Step the learning rate scheduler
            self.scheduler.step()

        self.total_time = time.time() - start_time

        
    def _run_loader(self, loader, is_training):
        running_loss = 0.0
        running_corrects = 0

        with torch.set_grad_enabled(is_training):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if is_training:
                    # Zero the optimizer gradients
                    self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimizer step if in training mode
                if is_training:
                    loss.backward()
                    self.optimizer.step()

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if is_training:
              self.scheduler.step()

        local_loss = torch.tensor( running_loss.item()).to(rank)
        local_correct = torch.tensor(running_corrects).to(rank)

        # Aggregate loss and accuracy across all GPUs
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)

        if rank == 0:  # Optionally, print on only one process
            # Normalize to get average loss and accuracy
            avg_loss = local_loss / dist.get_world_size()

        # return running_loss, running_corrects
        return avg_loss, local_correct


class PlantTrainerTuning(PlantTrainer):
    def __init__(self, *args, trial=None, **kwargs):
        self.trial = trial
        super().__init__(*args, **kwargs)
    
    # override
    def train_model(self):
        start_time = time.time()

        # Train the model for the specified number of epochs
        for epoch in range(self.num_epochs):
            epo_start_time = time.time()
            
            # Set the model to train mode
            self.model.train()
            # Training loop
            running_loss, running_corrects = self._run_loader(self.train_loader, is_training=True)

            # Calculate the train loss and accuracy
            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects / len(self.train_loader.dataset)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Set the model to evaluation mode
            self.model.eval()
            # Validation loop
            running_loss, running_corrects = self._run_loader(self.valid_loader, is_training=False)

            # Calculate the validation loss and accuracy
            valid_loss = running_loss / len(self.valid_loader.dataset)
            valid_acc = running_corrects / len(self.valid_loader.dataset)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_acc"].append(valid_acc)

            epo_elapsed_time = time.time() - epo_start_time
            self.history["epo_elapsed_time"].append(epo_elapsed_time)

            max_alloc = torch.cuda.max_memory_allocated(device=self.device) /  (1024 ** 2)  # in MB
            self.history["max_alloc"].append(max_alloc)

                        # Check if the current validation accuracy is the best so far
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                
            self.scheduler.step()

            # When compare the elapsed time, this pruned block will be commented
            if self.trial:
                self.trial.report(valid_acc, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            
        self.total_time = time.time() - start_time
