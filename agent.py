"""
agent.py  ─  Project Chitti v2
================================
Deep Q-Network (DQN) agent — updated for 8-dim observation space.

Architecture
────────────
  Input  : 8-dim observation  (ChittiEnv v2)
  Hidden : 2 × 128 ReLU layers
  Output : 6 Q-values

Training
────────
  Experience Replay  (ReplayBuffer, capacity 10 000)
  ε-greedy exploration  (ε: 1.0 → 0.05, decay 0.995 per learn step)
  Target network synced every 50 learn steps
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
#  Q-NETWORK
# ──────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    3-layer MLP approximating Q(s, a) for all actions simultaneously.

    Parameters
    ----------
    obs_dim    : int  – observation dimension (8 for Chitti v2)
    n_actions  : int  – number of discrete actions (6)
    hidden_dim : int  – neurons per hidden layer
    """

    def __init__(self, obs_dim: int = 8, n_actions: int = 6,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,    hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────
#  REPLAY BUFFER
# ──────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done) transitions.

    Parameters
    ----------
    capacity : int  – max stored transitions
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────
#  DQN AGENT
# ──────────────────────────────────────────────────────────────────

class ChittiAgent:
    """
    DQN agent that learns to operate Chitti's assistant + SOS logic.

    Parameters
    ----------
    obs_dim         : int    – 8 for Chitti v2
    n_actions       : int    – 6
    lr              : float  – Adam learning rate
    gamma           : float  – discount factor
    epsilon         : float  – initial exploration rate
    epsilon_min     : float  – minimum exploration rate
    epsilon_decay   : float  – multiplicative decay per learn step
    batch_size      : int    – mini-batch size
    target_update   : int    – learn-steps between target net syncs
    buffer_capacity : int    – replay buffer size
    """

    def __init__(
        self,
        obs_dim         : int   = 8,
        n_actions       : int   = 6,
        lr              : float = 1e-3,
        gamma           : float = 0.99,
        epsilon         : float = 1.0,
        epsilon_min     : float = 0.05,
        epsilon_decay   : float = 0.995,
        batch_size      : int   = 64,
        target_update   : int   = 50,
        buffer_capacity : int   = 10_000,
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.learn_steps   = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Online and target Q-networks
        self.q_net      = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(buffer_capacity)

        # Metrics
        self.loss_history   : list[float] = []
        self.reward_history : list[float] = []

    # ── Action selection ───────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        """
        ε-greedy: random with prob ε, else argmax Q(s,·).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_net(obs_t).argmax(dim=1).item())

    # ── Learning ───────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        """Add a transition to replay memory."""
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """
        One gradient update step.  Returns loss or None if not ready.
        """
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s  = torch.tensor(s,  device=self.device)
        a  = torch.tensor(a,  device=self.device)
        r  = torch.tensor(r,  device=self.device)
        ns = torch.tensor(ns, device=self.device)
        d  = torch.tensor(d,  device=self.device)

        q_curr = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next   = self.target_net(ns).max(1).values
            q_target = r + self.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay exploration
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str = "chitti_agent_v2.pth"):
        torch.save(self.q_net.state_dict(), path)
        print(f"[Agent] Saved → {path}")

    def load(self, path: str = "chitti_agent_v2.pth"):
        self.q_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"[Agent] Loaded ← {path}")