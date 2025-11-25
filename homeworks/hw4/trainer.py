import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

class ModelTrainer:
    
    """A training class for diffusion models."""
    
    def __init__(self, model, device, train_data, test_data, batch_size=1024, lr=1e-3,num_epochs=60,warmup_steps=100):
        """
        Initialize the trainer.
        
        Args:
            model: The diffusion model to train
            device: Device to run training on ('cuda' or 'cpu')
            train_data: Training data tensor
            test_data: Test data tensor
            batch_size: Batch size for training
            lr: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.test_data = test_data
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.total_steps = self.num_epochs * (len(train_data) // batch_size)
        self.test_losses = []
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []
        
        
    def lr_lambda(self,current_step):

        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps  # Linear warmup
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))  
    def evaluate_loss(self, data):
        """Evaluate the model on given data."""
        self.model.eval()
        total_loss = 0
        batch_size = 64
        n_batches = 0
        
        with torch.no_grad():
            for x in self.test_loader:
                x = x.to(self.device)
            # Sample t ~ U(0, 1)
                t = torch.rand(len(x), device=self.device)
                
                # Compute noise schedule
                alpha_t = torch.cos(np.pi * t / 2)
                sigma_t = torch.sin(np.pi * t / 2)
                
                # Sample noise and compute noisy input
                epsilon = torch.randn_like(x)
                x_t = alpha_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * epsilon
                
                # Get model prediction
                epsilon_pred = self.model(x_t, t)
                loss = torch.mean((epsilon - epsilon_pred) ** 2)
                
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        for x in self.train_loader:
            x = x.to(self.device)
            
            # Sample t ~ U(0, 1)
            t = torch.rand(len(x), device=self.device)
            
            # Compute noise schedule
            alpha_t = torch.cos(np.pi * t / 2)
            sigma_t = torch.sin(np.pi * t / 2)
            
            # Sample noise and compute noisy input
            epsilon = torch.randn_like(x)
            x_t = alpha_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * epsilon
            
            # Get model prediction and compute loss
            epsilon_pred = self.model(x_t, t)
            loss = torch.mean((epsilon - epsilon_pred) ** 2)
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Store training loss
            self.train_losses.append(loss.item())
    
    def train(self, print_every=10):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            print_every: Print test loss every N epochs
        """
        # Initial test loss
        initial_test_loss = self.evaluate_loss(self.test_data)
        self.test_losses.append(initial_test_loss)
        print(f'Initial test loss: {initial_test_loss:.4f}')
        
        for epoch in range(self.num_epochs):
            self.train_epoch()
            
            # Evaluate on test set
            test_loss = self.evaluate_loss(self.test_data)
            self.test_losses.append(test_loss)
            
            if (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch+1}/{self.num_epochs}, Test Loss: {test_loss:.4f}')
    
    def plot_losses(self):
        """Plot training and test losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='train loss', alpha=0.7)
        plt.plot(self.test_losses, label='test loss')
        plt.xlabel('Iteration / Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Losses')
        plt.legend()
        plt.grid(True)
        plt.show()
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    def get_losses(self):
        """Return training and test losses as numpy arrays."""
        return np.array(self.train_losses), np.array(self.test_losses)