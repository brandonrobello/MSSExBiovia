import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, MessagePassing, GINConv
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold


class BaselineGCN(nn.Module):
    def __init__(self, num_node_features, num_atom_types, hidden_dim=64):
        super().__init__()
        self.nn1 = GCNConv(num_node_features, hidden_dim)
        self.nn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_atom_types)

        self.pool = global_mean_pool

    def forward(self, data, return_graph_embedding=False):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.nn1(x, edge_index))
        x = F.relu(self.nn2(x, edge_index))

        if return_graph_embedding:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)  # Single-graph case
            graph_embedding = self.pool(x, batch)
            return graph_embedding  # Single vector per graph

        return self.fc(x)
    

class BaselineGAT(nn.Module):
    def __init__(self, num_node_features, num_atom_types, hidden_dim=64, heads=4):
        super().__init__()

        # GAT Layers
        self.nn1 = GATConv(num_node_features, hidden_dim, heads=heads)
        self.nn2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.fc = nn.Linear(hidden_dim, num_atom_types)  # Output layer

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index

        # GAT Layers with attention
        x = F.elu(self.nn1(x, edge_index))
        x = F.elu(self.nn2(x, edge_index))

        if return_embeddings:
            return x # Extract Embeddings
        
        x = self.fc(x)
        return x



class BaselineGIN(nn.Module):
    def __init__(self, num_node_features, num_atom_types, hidden_dim=512, dropout=0.2):
        super().__init__()

        # MLPs inside GIN
        self.mlp1 = nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.nn1 = GINConv(self.mlp1)
        self.nn2 = GINConv(self.mlp2)

        self.fc = nn.Linear(hidden_dim, num_atom_types)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        return self.fc(x)


class GCN_4Layer(nn.Module):
    def __init__(self, num_node_features, num_atom_types, hidden_dim=512):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2 )
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        self.conv4 = GCNConv(hidden_dim // 4, hidden_dim // 4)
        self.fc = nn.Linear(hidden_dim // 4, num_atom_types)

        self.pool = global_mean_pool

    def forward(self, data, return_graph_embedding=False):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))


        if return_graph_embedding:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)  # Single-graph case
            graph_embedding = self.pool(x, batch)
            return graph_embedding  # Single vector per graph

        return self.fc(x)
    

class GAT_4L(nn.Module):
    def __init__(self, num_node_features, num_atom_types, hidden_dim=1024, num_heads=4, dropout=0.2):
        super().__init__()
        
        # GAT Layers (multi-head attention)
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=num_heads)
        self.conv4 = GATConv(hidden_dim // 2 * num_heads, hidden_dim // 4, heads=1, concat=False)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim // 4, num_atom_types)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GAT Layers with ELU
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))

        return self.fc(x)  # Predict atom types (logits)

# Next to implement
# GraphSAGE (SAGEConv) → Learns representations from neighbor aggregation.
# Graph Attention Networks (GAT) → Uses attention mechanisms to weigh neighbor contributions.
# Message Passing Neural Networks (MPNN) → More expressive graph representation.
# Cross-validation
# Dropout
# normalization


# TRAINER
class GNNTrainer:
    def __init__(self, model, dataset, batch_size=32, learning_rate=0.001, \
                 epochs=20, k_folds=5, task="classification", random_seed=21):
        """
        Initializes the GNN trainer.

        Args:
            model (nn.Module): The GNN model to train.
            dataset (Dataset): Full dataset (train + validation).
            batch_size (int): Batch size for DataLoader.
            learning_rate (float): Learning rate for optimizer.
            epochs (int): Number of training epochs.
            task (str): "classification" or "regression" (determines loss function).
            k_folds (int): Number of folds for cross-validation.

        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.task = task
        self.random_seed = random_seed
        self.k_folds = k_folds
        self.k_fold_splits = {}

        # Initialize optimizer & loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

        # Print model summary
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model.__class__.__name__} - Number of parameters: {num_params}")

    def train(self, draw_curve=True, quiet_mode=False):
        """
        Train the GNN model and plot training loss.

        Args:
            draw_curve (bool): If True, plot loss curve.

        Returns:
            List of batch losses over training.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_seed)

        train_losses = []
        val_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            # Create train & validation loaders for this fold
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            # Store loss per epoch for this fold
            train_loss_per_epoch = []
            val_loss_per_epoch = []

            for epoch in tqdm(range(self.epochs), leave=False):
                self.model.train()
                epoch_loss = 0

                total_batch_loss = 0.0
                total_samples = 0  # Keep track of total samples processed  
                for batch_data in train_loader:
                    batch_data = batch_data.to(next(self.model.parameters()).device)  # Ensure device consistency
                    batch_size = batch_data.y.size(0)  # Get actual batch size

                    
                    self.optimizer.zero_grad()

                    # Forward pass
                    batch_pred = self.model(batch_data)

                    # Compute loss
                    batch_loss = self.loss_fn(batch_pred, batch_data.y)
                    batch_loss.backward()
                    self.optimizer.step()

                    
                    # Normalize by batch size and accumulate
                    total_batch_loss += batch_loss.item() * batch_size  # Scale loss by batch size
                    total_samples += batch_size  # Track total number of samples processed

                epoch_loss += total_batch_loss/ total_samples
                train_loss_per_epoch.append(epoch_loss)

                # Evaluate Validation Loss
                val_loss = self.evaluate(val_loader, print_loss=False)
                val_loss_per_epoch.append(val_loss)

            if not quiet_mode:
                print(f"---------KFOLD: {fold}----------")
                i = 0
                for epoch_loss, val_loss in zip(train_loss_per_epoch, val_loss_per_epoch):
                    i += 1
                    if i % 10 == 0:
                        print(f"Epoch - {i}: Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Store losses for this fold
            train_losses.append(train_loss_per_epoch)
            val_losses.append(val_loss_per_epoch)

            self.k_fold_splits[fold] = (train_idx, val_idx)

        # Plot Loss Curves
        if draw_curve:
            self.plot_loss(train_losses, val_losses)


        return {
            "train_loss_list": train_losses,
            "val_loss_list": val_losses
        }
        
    def evaluate(self, dataloader, print_loss=True):
        """
        Evaluate model on a given dataset.

        Args:
            data_loader (DataLoader): Validation/test data loader.
            print_loss (bool): Whether to print loss.

        Returns:
            Total loss on the dataset.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(next(self.model.parameters()).device)  # Move to correct device
                batch_pred = self.model(batch_data)
                batch_loss = self.loss_fn(batch_pred, batch_data.y)
                total_loss += batch_loss.item() /len(dataloader)
        if print_loss:
            print(f"Validation Loss: {total_loss:.4f}")
        return total_loss
    
    def plot_loss(self, train_losses, val_losses):
        """
        Plots the training and validation loss per fold and averages across k-folds.

        Args:
            train_losses (list): List of lists containing training loss per epoch for each fold.
            val_losses (list): List of lists containing validation loss per epoch for each fold.
        """
        num_folds = len(train_losses)
        epochs = len(train_losses[0])

        plt.figure(figsize=(10, 5))

        # === (1) Plot Each Fold Individually ===
        for fold in range(num_folds):
            plt.plot(train_losses[fold], linestyle="dotted", alpha=0.7, label=f"Train Fold {fold+1}")
            plt.plot(val_losses[fold], linestyle="dashed", alpha=0.7, label=f"Val Fold {fold+1}")

        # === (2) Compute & Plot Mean/Std Across Folds ===
        mean_train_loss = np.mean(train_losses, axis=0)
        std_train_loss = np.std(train_losses, axis=0)

        mean_val_loss = np.mean(val_losses, axis=0)
        std_val_loss = np.std(val_losses, axis=0)

        plt.plot(mean_train_loss, color="blue", label="Mean Train Loss")
        plt.fill_between(range(epochs), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2, color="blue")

        plt.plot(mean_val_loss, color="red", label="Mean Validation Loss")
        plt.fill_between(range(epochs), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.2, color="red")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Across K-Folds")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_model(self, dataset):
        """
        Evaluate the GNN model on dataset and print accuracy or MSE.

        Returns:
            dict: Contains evaluation metrics (Accuracy, F1-score for classification, MSE for regression).
        """
        self.model.eval()
        y_true, y_pred = [], []

        eval_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():
            for batch_data in tqdm(eval_loader, leave=False):
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
    
    def predict(self, dataset):
        """
        Predicts atom types on a new dataset (X_new) or the test dataset.

        Args:
            X_new (Dataset, optional): A new dataset to predict on. If None, runs on test dataset.

        Returns:
            Tuple[List[str], List[str]]: Predicted and true atom types as string labels.
        """
        self.model.eval()
        y_true, y_pred = [], []
        pred_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():
            for batch_data in tqdm(pred_loader, leave=False):
                batch_data = batch_data.to(next(self.model.parameters()).device)
                batch_pred = self.model(batch_data)

                y_pred.extend(batch_pred.cpu().numpy())  # Collect predictions
                y_true.extend(batch_data.y.cpu().numpy().flatten())  # Collect true labels

        # Convert to NumPy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Convert predictions to class indices
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Fetch the original dataset, not the Subset!
        original_dataset = getattr(self.dataset, "dataset", self.dataset)
        label_encoder = getattr(original_dataset, "label_encoder", None)

        # Convert indices to atom type names using stored mapping
        if label_encoder:
            y_pred_labels = label_encoder.inverse_transform(y_pred_classes.tolist())
            y_true_labels = label_encoder.inverse_transform(y_true.tolist())
        else:
            y_pred_labels = y_pred_classes.tolist()
            y_true_labels = y_true.tolist()

        return y_true_labels, y_pred_labels