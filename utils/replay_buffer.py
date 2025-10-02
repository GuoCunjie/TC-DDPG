import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False
        self.s = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.r = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, s, a, r):
        n = s.shape[0]
        for i in range(n):
            self.s[self.ptr] = s[i]
            self.a[self.ptr] = a[i]
            self.r[self.ptr,0] = r[i]
            self.ptr = (self.ptr + 1) % self.capacity
            if self.ptr == 0: self.full = True

    def sample(self, batch):
        size = self.capacity if self.full else self.ptr
        idx = np.random.randint(0, size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx]
