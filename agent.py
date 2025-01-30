import torch
import torch.nn as nn
import torch.optim as optim
from memory import ReplayBuffer
from model import QNetwork
import math
import random


class DQNAgent():
    def __init__ (self,
                  env,
                  device,
                  batch_size=128,
                  gamma=0.99,
                  tau=0.005,
                  lr=1e-4,
                  memory_length=10000,
                  eps_start=0.9,
                  eps_end=0.05,
                  eps_decay=1000):

        self.env = env
        self.state_n = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.memory_length = memory_length
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.policy_net = QNetwork(self.state_n, self.action_n).to(self.device)
        self.target_net = QNetwork(self.state_n, self.action_n).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(memory_length, self.device)

        self.steps_done = 0
        self.eps_threshold = None
        self.update_eps_th()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if random.random() > self.eps_threshold:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

        self.steps_done += 1
        self.update_eps_th()

        return action

    def learn(self, state, action, reward, next_state):

        self.memory.push(
            torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
            action,
            torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0) if next_state is not None else next_state,
            torch.tensor([reward], dtype=torch.float32, device=self.device)
            )

        if len(self.memory) > self.batch_size:
            states, actions, rewards, non_final_next_states, non_final_mask = self.memory.sample(self.batch_size)

            q_values = self.policy_net(states).gather(1, actions)

            next_state_values = torch.zeros(self.batch_size).to(self.device)

            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

            target_q_values = (next_state_values * self.gamma) + rewards

            loss = self.criterion(q_values, target_q_values.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

            self.soft_update_weights(self.policy_net, self.target_net)

    def soft_update_weights(self, policy_net, target_net):

        policy_net_state_dict = policy_net.state_dict()
        target_net_state_dict = target_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        target_net.load_state_dict(target_net_state_dict)

    def update_eps_th(self):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)