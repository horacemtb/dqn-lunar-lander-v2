import torch
import numpy as np
from agent import DQNAgent
import os


def running_mean(x, N=100):
    x = np.array(x)
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


def save_agent(agent, folder='model_weights', filename='dqn_agent'):

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    torch.save(agent.policy_net.state_dict(), f"{filepath}.pth")
    print(f'Agent saved to {filepath}')


def load_agent(agent, filepath='model_weights/dqn_agent.pth', device='cpu'):

    agent.policy_net.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
    print("Agent's policy loaded")