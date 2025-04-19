import torch
import numpy as np

class Databatch:
    def __init__(self, batch_size, state_dim, action_dim):
        self.state = np.zeros((batch_size, state_dim))
        self.action = np.zeros((batch_size, action_dim))
        self.logprob = np.zeros((batch_size, action_dim))
        self.reward = np.zeros((batch_size, 1))
        self.next_sate = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, state, action, logprob, reward, next_state, dw, done):
        self.state[self.count] = state
        self.action[self.count] = action
        self.logprob[self.count] = logprob
        self.reward[self.count] = reward
        self.next_sate[self.count] = next_state
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        state = torch.tensor(self.state, dtype=torch.float)
        action = torch.tensor(self.action, dtype=torch.float)
        logprob = torch.tensor(self.logprob, dtype=torch.float)
        reward = torch.tensor(self.reward, dtype=torch.float)
        next_sate = torch.tensor(self.next_sate, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return state, action, logprob, reward, next_sate, dw, done
