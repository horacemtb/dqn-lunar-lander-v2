import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, fc1_size=64, fc2_size=128):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(n_observations, fc1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_size, n_actions)

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)