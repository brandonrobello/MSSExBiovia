import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from typing import Optional, Union, Dict, Any

from atoMLtype.models.BaseGNNModel import BaseGNNModel
from atoMLtype.datasets.BaseDataset import BaseDataset


class GNNTrainer:
    """
    Trainer for training and evaluating GNN models on atom-type classification tasks.

    Supports both standard training and k-fold cross-validation.

    Args:
        model (BaseGNNModel): The GNN model to train.
        dataset (Dataset): PyTorch Geometric dataset of molecular graphs.
        batch_size (int): Batch size for training.
        learning_rate (float): Optimizer learning rate.
        epochs (int): Number of epochs to train.
        k_folds (int or None): Number of folds for cross-validation.
        random_seed (int): Seed for reproducibility.
    """
    def __init__(self, 
             model: BaseGNNModel, 
             dataset: BaseDataset, 
             batch_size: int = 32, 
             learning_rate: float = 1e-3,
             epochs: int = 20, 
             k_folds: Optional[int] = None, 
             random_seed: int = 42):
        
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds
        self.random_seed = random_seed
        self.k_fold_splits = {}

        # Initialize optimizer & loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        # Print model summary
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model.__class__.__name__} - Number of parameters: {num_params}")

    def train(self, 
              draw_curve: bool = True, 
              verbose: bool = False, 
              report_step: int = 5, 
              force: bool = False) -> Union[Dict[str, Any], None]:
        """
        Train the model using either standard training or k-fold cross-validation.

        Args:
            draw_curve (bool): Whether to plot the loss curve after training.
            verbose (bool): Whether to print loss during training.
            report_step (int): How often to print loss (in epochs).
            force (bool): Force retraining even if the model is marked trained.

        Returns:
            dict: Training loss and optionally validation loss per epoch/fold.
        """
        if not force and self.model.is_trained():
            raise RuntimeError("Model is already trained. Use `force=True` to retrain.")

        if self.k_folds and self.k_folds > 1:
            loss_output = self._train_kfold(draw_curve, verbose, report_step)
        else:
            loss_output = self._train_standard(draw_curve, verbose, report_step)

        self.model.set_trained()
        return loss_output
    

    def _train_standard(self, draw_curve=True, verbose=False, report_step=5):
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        train_loss_per_epoch = []

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            self.model.train()
            total_loss = 0
            total_samples = 0

            for batch_data in train_loader:
                batch_data = batch_data.to(next(self.model.parameters()).device)
                output = self.model(batch_data)
                logits = output.logits
                loss = self.loss_fn(logits, batch_data.y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_data.y.size(0)
                total_samples += batch_data.y.size(0)

            epoch_loss = total_loss / total_samples
            train_loss_per_epoch.append(epoch_loss)
        
        if verbose:
            for epoch in range(0, self.epochs, report_step):
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss_per_epoch[epoch]:.4f}")

        if draw_curve:
            self._plot_loss(train_loss_per_epoch)

        return {"train_loss": train_loss_per_epoch}

    def _train_kfold(self, draw_curve=True, verbose=False, report_step=5):

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_seed)
        train_losses, val_losses = [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            print(f"--- Fold {fold + 1}/{self.k_folds} ---")

            train_loader = DataLoader(Subset(self.dataset, train_idx), batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(Subset(self.dataset, val_idx), batch_size=self.batch_size, shuffle=False)

            train_loss_per_epoch = []
            val_loss_per_epoch = []

            for epoch in tqdm(range(self.epochs), leave=False):
                self.model.train()
                total_loss = 0
                total_samples = 0

                for batch_data in train_loader:
                    batch_data = batch_data.to(next(self.model.parameters()).device)
                    logits = self.model(batch_data).logits
                    loss = self.loss_fn(logits, batch_data.y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() * batch_data.y.size(0)
                    total_samples += batch_data.y.size(0)

                epoch_loss = total_loss / total_samples
                train_loss_per_epoch.append(epoch_loss)

                val_loss = self._evaluate_loss(val_loader)
                val_loss_per_epoch.append(val_loss)

            if verbose:
                for epoch in range(0, self.epochs, report_step):
                    print(f"Epoch {epoch + 1}: Train Loss = {train_loss_per_epoch[epoch]:.4f}, Val Loss = {val_loss_per_epoch[epoch]:.4f}")

            self.k_fold_splits[fold] = (train_idx, val_idx)
            train_losses.append(train_loss_per_epoch)
            val_losses.append(val_loss_per_epoch)

        if draw_curve:
            self._plot_kfold_loss(train_losses, val_losses)

        return {"train_loss": train_losses, "val_loss": val_losses}

    def _evaluate_loss(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(next(self.model.parameters()).device)
                logits = self.model(batch).logits
                loss = self.loss_fn(logits, batch.y)

                total_loss += loss.item() * batch.y.size(0)
                total_samples += batch.y.size(0)

        return total_loss / total_samples

    def _plot_loss(self, train_loss):
        plt.plot(train_loss, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    def _plot_kfold_loss(self, train_losses, val_losses):
        num_folds = len(train_losses)
        epochs = len(train_losses[0])
        plt.figure(figsize=(10, 5))

        # Fold-wise losses
        for i in range(num_folds):
            plt.plot(train_losses[i], linestyle="--", alpha=0.6, label=f"Train Fold {i+1}")
            plt.plot(val_losses[i], linestyle="-.", alpha=0.6, label=f"Val Fold {i+1}")

        # Mean + std
        train_mean = np.mean(train_losses, axis=0)
        train_std = np.std(train_losses, axis=0)
        val_mean = np.mean(val_losses, axis=0)
        val_std = np.std(val_losses, axis=0)

        plt.plot(train_mean, label="Mean Train", color="blue")
        plt.fill_between(range(epochs), train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")

        plt.plot(val_mean, label="Mean Val", color="red")
        plt.fill_between(range(epochs), val_mean - val_std, val_mean + val_std, alpha=0.2, color="red")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("K-Fold Cross-Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.show()
