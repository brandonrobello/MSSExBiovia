import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from atoMLtype.RF.metrics import plot_full_confusion_matrix


class BaselineGNN(nn.Module):
    def __init__(self, num_node_features, num_atom_types, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_atom_types)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)  # Output logits for classification
        return x
    
class GNNWithEmbeddings(nn.Module):
    def __init__(self, num_node_features, num_atom_types, embedding_dim=16, hidden_dim=64):
        super().__init__()

        # Atom Type Embedding
        self.atom_embedding = nn.Embedding(num_atom_types, embedding_dim)

        # GNN Layers
        self.conv1 = GCNConv(num_node_features + embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_atom_types)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Embed atom types and concatenate with node features
        atom_embeds = self.atom_embedding(data.y)  # Shape: [num_atoms, embedding_dim]
        x = torch.cat([x, atom_embeds], dim=1)  # Shape: [num_atoms, feature_dim + embedding_dim]

        # GNN Layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)  # Output logits

        return x

class GNNWithGAT(nn.Module):
    def __init__(self, num_node_features, num_atom_types, embedding_dim=16, hidden_dim=64, heads=4):
        super().__init__()

        # Atom Type Embedding
        self.atom_embedding = nn.Embedding(num_atom_types, embedding_dim)

        # GAT Layers
        self.conv1 = GATConv(num_node_features + embedding_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.fc = nn.Linear(hidden_dim, num_atom_types)  # Output layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Embed atom types and concatenate with node features
        atom_embeds = self.atom_embedding(data.y)
        x = torch.cat([x, atom_embeds], dim=1)

        # GAT Layers with attention
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.fc(x)

        return x

# Next to implement
# GraphSAGE (SAGEConv) → Learns representations from neighbor aggregation.
# Graph Attention Networks (GAT) → Uses attention mechanisms to weigh neighbor contributions.
# Message Passing Neural Networks (MPNN) → More expressive graph representation.
# Cross-validation
# Dropout
# normalization


# TRAINER

class GNNTrainer:
    def __init__(self, model, train_dataset, test_dataset, batch_size=32, learning_rate=0.001, epochs=20, task="classification"):
        """
        Initializes the GNN trainer.

        Args:
            model (nn.Module): The GNN model to train.
            train_dataset (Dataset): Training dataset.
            test_dataset (Dataset): Testing dataset.
            batch_size (int): Batch size for DataLoader.
            learning_rate (float): Learning rate for optimizer.
            epochs (int): Number of training epochs.
            task (str): "classification" or "regression" (determines loss function).
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.task = task

        # Create DataLoaders internally
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize optimizer & loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
        batch_loss_list = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            print(f"Epoch: {epoch+1}/{self.epochs}")
            
            for batch_data in tqdm(self.train_loader, leave=False):
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

    def evaluate(self):
        """
        Evaluate the GNN model on the test set and print accuracy or MSE.

        Returns:
            dict: Contains evaluation metrics (Accuracy, F1-score for classification, MSE for regression).
        """
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch_data in tqdm(self.test_loader, leave=False):
                batch_data = batch_data.to(next(self.model.parameters()).device)
                batch_pred = self.model(batch_data)

                y_pred.extend(batch_pred.cpu().numpy())  # Collect predictions
                y_true.extend(batch_data.y.cpu().numpy().flatten())  # Collect true labels

        # Ensure y_true and y_pred are NumPy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {}

        if self.task == "classification":
            y_pred_classes = np.argmax(y_pred, axis=1)  # Convert logits to class indices

            # Compute classification metrics
            acc = accuracy_score(y_true, y_pred_classes)
            f1 = f1_score(y_true, y_pred_classes, average="weighted")
            metrics = {"Accuracy": acc, "F1-score": f1}

            print(f"Evaluation Metrics: {metrics}")

        elif self.task == "regression":
            mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
            metrics = {"MSE": mse}
            print(f"Evaluation Metrics: {metrics}")

        return metrics
    
    def predict(self, X_new=None):
        """
        Predicts atom types on a new dataset (X_new) or the test dataset.

        Args:
            X_new (Dataset, optional): A new dataset to predict on. If None, runs on test dataset.

        Returns:
            Tuple[List[str], List[str]]: Predicted and true atom types as string labels.
        """
        self.model.eval()
        y_true, y_pred = [], []
        dataset = X_new if X_new else self.test_loader

        with torch.no_grad():
            for batch_data in tqdm(dataset, leave=False):
                batch_data = batch_data.to(next(self.model.parameters()).device)
                batch_pred = self.model(batch_data)

                y_pred.extend(batch_pred.cpu().numpy())  # Collect predictions
                y_true.extend(batch_data.y.cpu().numpy().flatten())  # Collect true labels

        # Convert to NumPy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Convert predictions to class indices
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Extract atom_type_mapping if it exists in the model
        atom_type_mapping = getattr(self.model, "atom_type_mapping", None)

        # Fetch the original dataset, not the Subset!
        original_dataset = getattr(self.train_loader.dataset, "dataset", self.train_loader.dataset)
        label_encoder = getattr(original_dataset, "label_encoder", None)

        # Convert indices to atom type names using stored mapping
        if atom_type_mapping:
            y_pred_labels = label_encoder.inverse_transform([self.atom_type_mapping[idx] for idx in y_pred_classes])
            y_true_labels = label_encoder.inverse_transform([self.atom_type_mapping[idx] for idx in y_true])
        else:
            y_pred_labels = label_encoder.inverse_transform(y_pred_classes.tolist())
            y_true_labels = label_encoder.inverse_transform(y_true.tolist())

        return y_true_labels, y_pred_labels