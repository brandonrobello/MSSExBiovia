import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

class BaselineGNN(nn.Module):
    def __init__(self, num_node_features, num_atom_types, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_atom_types)  # Classification output

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)  # Output logits for classification
        return x

class GNNTrainer:
    def __init__(self, model, dataset, batch_size=32, learning_rate=0.001, epochs=10, l2=1e-4, task="regression"):
        """
        A general trainer class for GNN models.

        Args:
            model (torch.nn.Module): The GNN model to train.
            dataset (Dataset): The dataset to use for training.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimizer.
            epochs (int): Number of training epochs.
            l2 (float): L2 weight decay.
            task (str): "regression" or "classification" (determines loss function).
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.task = task

        # Initialize optimizer & loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
        self.loss_fn = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

        # Print model summary
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model.__class__.__name__} - Number of parameters: {num_params}")

    def train(self, draw_curve=True):
        """
        Train the GNN model and plot training loss.

        Args:
            draw_curve (bool): If True, plot loss curve.

        Returns:
            List of batch losses over training.
        """
        self.model.train()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        batch_loss_list = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            print(f"Epoch: {epoch+1}/{self.epochs}")
            
            for batch_data in tqdm(loader, leave=False):
                batch_data = batch_data.to(next(self.model.parameters()).device)  # Ensure device consistency
                self.optimizer.zero_grad()

                # Forward pass
                batch_pred = self.model(batch_data)

                # Compute loss
                batch_loss = self.loss_fn(batch_pred, batch_data.y)
                batch_loss.backward()
                self.optimizer.step()

                batch_loss_list.append(batch_loss.item())
                epoch_loss += batch_loss.item()

            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

        if draw_curve:
            plt.figure(figsize=(5, 4))
            plt.plot(batch_loss_list, label="Training Loss")
            plt.yscale("log")
            plt.xlabel("# Batch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        return batch_loss_list

    def evaluate(self, draw_curve=True):
        """
        Evaluate the GNN model and compute the appropriate metrics.

        Args:
            draw_curve (bool): If True, plot prediction results.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=self.batch_size)
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch_data in tqdm(loader, leave=False):
                batch_data = batch_data.to(next(self.model.parameters()).device)
                batch_pred = self.model(batch_data)

                y_pred.append(batch_pred.cpu().numpy())  # Do not flatten yet
                y_true.append(batch_data.y.cpu().numpy().flatten())

        # Concatenate all batches correctly
        y_true = np.concatenate(y_true, axis=0)  # Ensure proper shape
        y_pred = np.concatenate(y_pred, axis=0)

        if self.task == "classification":
            if y_pred.ndim == 1:  # Handle incorrect shape case
                y_pred_classes = y_pred.astype(int)
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)

            acc = accuracy_score(y_true, y_pred_classes)
            f1 = f1_score(y_true, y_pred_classes, average="weighted")
            metrics = {"Accuracy": acc, "F1-score": f1}
        else:
            mse = mean_squared_error(y_true, y_pred)
            metrics = {"MSE": mse}

        print(f"Evaluation Metrics: {metrics}")

        if draw_curve:
            plt.figure(figsize=(5, 4))
            if self.task == "regression":
                plt.scatter(y_true, y_pred, label=f"MSE: {metrics['MSE']:.2f}", s=2)
                plt.xlabel("Ground Truth")
                plt.ylabel("Predicted")
                plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
            else:
                plt.hist(y_pred_classes - y_true, bins=np.arange(-1.5, 1.5, 0.5), alpha=0.7, label="Prediction Errors")
                plt.xlabel("Prediction Error")
                plt.ylabel("Count")

            plt.legend()
            plt.show()

        return metrics
        