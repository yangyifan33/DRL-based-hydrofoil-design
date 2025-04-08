import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x))
        return mean
    
    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

class Agent():
    def __init__(self, state_dim, action_dim, gamma, lamda, batch_size, mini_batch_size, lr_a, lr_c, epochs, epsilon, episode_num):
        
        self.lr_a = lr_a
        self.lr_c = lr_c

        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr_c, eps=1e-5)

        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.episode_num = episode_num

        self.max_action = torch.tensor([0.75, 0.02]).view(1,2)
        self.min_action = torch.tensor([-0.75, -0.02]).view(1,2)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.policy_net.get_dist(state)
            action = dist.sample()  # Sample the action according to the probability distribution
            a_logprob = dist.log_prob(action)  # The log probability density of the action

        return action.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, batch, episode):
        state, action, log_prob, reward, next_state, dw, done = batch.numpy_to_tensor()  # Get training data
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            values = self.value_net(state)
            next_values = self.value_net(next_state)
            deltas = reward + self.gamma * (1.0 - dw) * next_values - values
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + values
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for epochs:
        for _ in range(self.epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.policy_net.get_dist(state[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                logprob_now = dist_now.log_prob(action[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(logprob_now.sum(1, keepdim=True) - log_prob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                policy_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
                # Update policy_net
                self.optimizer_policy.zero_grad()
                policy_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer_policy.step()

                values = self.value_net(state[index])
                value_loss = F.mse_loss(v_target[index], values)
                # Update value_net
                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer_value.step()
        self.lr_decay(episode)

    def lr_decay(self, episode):
        lr_a_now = self.lr_a * (1 - episode / self.episode_num)
        lr_c_now = self.lr_c * (1 - episode / self.episode_num)
        for p in self.optimizer_policy.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_value.param_groups:
            p['lr'] = lr_c_now

    def lr_reset(self):
        for p in self.optimizer_policy.param_groups:
            p['lr'] = self.lr_a
        for p in self.optimizer_value.param_groups:
            p['lr'] = self.lr_c
    
    def episodes_reset(self, episode_num):
        self.episode_num = episode_num