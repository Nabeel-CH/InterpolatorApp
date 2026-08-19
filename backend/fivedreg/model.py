"""Model definition for the neural network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP)
    Given an input dimension, a list of hidden units, and an output dimension,
    builds a feedforward neural network.

    Attributes:
        model (nn.Sequential): A sequential container of Linear + activation
            layers that defines the MLP architecture.
    """

    def __init__(self, input_dim, hidden_units, output_dim=1):
        """
        Initialize the MLP

        Parameters:
            input_dim (int): Number of input features
            hidden_units (list[int]): Number of units in each hidden layer.
            output_dim (int): Number of output units.
        """
        super().__init__()

        layers = []

        # build hidden layers
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units

        # final layer gives a single output value
        layers.append(nn.Linear(input_dim, output_dim))

        self.model = nn.Sequential(*layers)

    # forward method
    def forward(self, x):
        """
        Run a forward pass through the model

        Parameters:
            x (torch.Tensor): Input tensor 
    
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

class NeuralNetwork:
    """
    Feedforward neural network regressor.
    Wraps a MLP model and has a fit + predict method 
    """

    def __init__(
        self,
        input_dim=5,
        hidden_layers=None,
        learning_rate=1e-3,
        max_epochs=200,
        batch_size=64,
    ):
        """
        Initialize the NeuralNetwork.

        Parameters:
            input_dim (int): Number of input features.
            hidden_layers (list[int]): Number of units in each hidden layer.
            learning_rate (float): Learning rate.
            max_epochs (int): Number of training epochs.
            batch_size (int): Batch size.
        """
        # default architecture if hidden_layers is not passed in
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # use CPU
        self.device = torch.device("cpu")

        # create the neural network
        self.net = MLP(input_dim, hidden_layers, output_dim=1).to(self.device)

        # loss and optimizer function
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        """
        Train the network 

        Parameters:
            X (array): Training features
            y (array): Training targets

        Returns:
            NeuralNetwork: The fitted model.
        """
        # make sure everything is in the right format
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.net.train()

        for epoch in range(self.max_epochs):
            total_loss = 0.0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # forward pass
                preds = self.net(batch_X)
                loss = self.loss_fn(preds, batch_y)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(dataset)
            # to see training progress
            print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

        return self

    def predict(self, X):
        """
        Run the trained network on new data and return predictions.

        Parameters:
            X (array): Input features

        Returns:
            numpy.ndarray: Predicted values
        """
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X).to(self.device)

        self.net.eval()
        with torch.no_grad():
            preds = self.net(X_tensor).cpu().numpy().reshape(-1)

        return preds
