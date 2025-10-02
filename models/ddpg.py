import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim=256, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )
        self.bounds = {
            "irrigation": (0.5, 8.0),
            "fertilizer": (2.0, 20.0),
            "ventilation": (0.0, 24.0),
            "light_supplement": (0.0, 200.0)
        }

    def forward(self, s):  # [B, state_dim]
        x = self.net(s)  # logits
        b = self.bounds
        a1 = (torch.tanh(x[:,0:1])+1)/2 * (b["irrigation"][1]-b["irrigation"][0]) + b["irrigation"][0]
        a2 = (torch.tanh(x[:,1:2])+1)/2 * (b["fertilizer"][1]-b["fertilizer"][0]) + b["fertilizer"][0]
        a3 = (torch.tanh(x[:,2:3])+1)/2 * (b["ventilation"][1]-b["ventilation"][0]) + b["ventilation"][0]
        a4 = (torch.tanh(x[:,3:4])+1)/2 * (b["light_supplement"][1]-b["light_supplement"][0]) + b["light_supplement"][0]
        return torch.cat([a1,a2,a3,a4], dim=1)

class Critic(nn.Module):
    def __init__(self, state_dim=256, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s,a], dim=1))

class DDPG(nn.Module):
    def __init__(self, state_dim=256, action_dim=4):
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_targ = Actor(state_dim, action_dim)
        self.critic_targ = Critic(state_dim, action_dim)
        self._hard_update(self.actor_targ, self.actor)
        self._hard_update(self.critic_targ, self.critic)

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @torch.no_grad()
    def soft_update(self, tau: float):
        for t, s in zip(self.actor_targ.parameters(), self.actor.parameters()):
            t.data.mul_(1-tau).add_(tau*s.data)
        for t, s in zip(self.critic_targ.parameters(), self.critic.parameters()):
            t.data.mul_(1-tau).add_(tau*s.data)
