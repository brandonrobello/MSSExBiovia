import pickle
import datetime
import os
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class BaseModel(ABC):
    """Abstract base class for machine learning models."""
    
    @abstractmethod
    def train(self, X, y, test_size=0.2, random_state=21):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_model(self, save_dir="saved_models"):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass


class RandomForestModel(BaseModel):
    """
    RandomForest model with label encoding, training, prediction, test dataset management,
    and model serialization.
    
    Args:
        n_estimators (int): Number of trees in the forest.
    """
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=21)
        self.label_encoder = LabelEncoder()
        self.is_trained = False  # Track training state
        self.X_train = None  # Store training feature matrix
        self.y_train = None  # Store training labels

    def train(self, X, y, test_size=0.2, random_state=21, force_load=False):
        """
        Trains the RandomForest model with label encoding 
        and train-test split. Stores training data for reproducibility.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.
            y (list or np.array): Target labels.
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
            force_load (bool): If True, allows overwriting an already trained model.
        
        Returns:
            tuple: (X_test, y_test_decoded) - Test dataset after train-test split.
        """
        if self.is_trained and not force_load:
            raise ValueError(
                "Warning: Model is already trained. If you want to continue, retry with force_load=True."
            )

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )

        # Train model
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        print("Model trained successfully.")

        return self.X_test.copy(), self.label_encoder.inverse_transform(self.y_test)
    
    def predict(self, X):
        """
        Predicts labels for new data.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.
        
        Returns:
            np.array: Predicted labels (decoded).
        """
        if not self.is_trained:
            raise ValueError("The model has not been trained or loaded. Train or load a model first.")

        y_pred_encoded = self.model.predict(X)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred_decoded

    def save_model(self, save_dir="saved_models"):
        """
        Serializes using pickle and saves the trained model with a timestamp.
        
        Args:
            save_dir (str): Directory where the model is saved.
        """
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{save_dir}/RandomForestModel_{timestamp}.pkl"

        # Save model, label encoder, and training data
        with open(model_filename, "wb") as file:
            pickle.dump(
                {"model": self.model, "label_encoder": self.label_encoder, 
                 "X_train": self.X_train, "y_train": self.y_train}, 
                file
            )


        print(f"Model saved as: {model_filename}")

    def load_model(self, model_path, force_load=False):
        """
        Loads a previously saved model.

        Args:
            model_path (str): Path to the saved model file.
            force_load (bool): If True, allows overwriting an already trained model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please provide a valid path.")

        if self.is_trained and not force_load:
            raise ValueError(
                "Warning: Model is already trained. If you want to continue, retry with force_load=True."
            )
        
        with open(model_path, "rb") as file:
            saved_data = pickle.load(file)

        self.model = saved_data["model"]
        self.label_encoder = saved_data["label_encoder"]
        self.X_train = saved_data.get("X_train", None)
        self.y_train = saved_data.get("y_train", None)
        self.is_trained = True
        print(f"Model loaded from {model_path}.")


