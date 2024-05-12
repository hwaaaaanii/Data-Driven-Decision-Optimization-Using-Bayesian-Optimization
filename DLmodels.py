import torch
import torch.nn as nn
import torch.optim as optim

# Define a Simple Neural Network
class BaseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(BaseMLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.model(x)

# Define an Improved Neural Network
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ImprovedMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),                nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.model(x)

    # Define a Complex Neural Network
class ComplexMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(ComplexMLP, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
