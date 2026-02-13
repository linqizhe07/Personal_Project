"""
HILP (Hilbert Foundation Policy) feature network and intrinsic reward.

The feature network xi(s) maps state s -> R^d where d is the feature dimension.
The intrinsic reward is: r_int = (xi(s') - xi(s))^T z
where z is a skill vector sampled uniformly from the unit ball.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class HILPFeatureNetwork(nn.Module):
    """
    Temporal distance feature network xi(s).

    Architecture: MLP with layer normalization.
    Input: observation (obs_dim,)
    Output: feature vector (feature_dim,), L2-normalized to unit sphere.
    """

    def __init__(
        self,
        obs_dim: int = 376,
        feature_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)
        return nn.functional.normalize(raw, dim=-1)


def compute_intrinsic_reward(
    xi_s: torch.Tensor,
    xi_s_next: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Intrinsic reward: r_int = (xi(s') - xi(s))^T z

    Args:
        xi_s: (batch, feature_dim) features of current state
        xi_s_next: (batch, feature_dim) features of next state
        z: (batch, feature_dim) or (feature_dim,) skill vector
    Returns:
        reward: (batch,)
    """
    delta_xi = xi_s_next - xi_s
    return (delta_xi * z).sum(dim=-1)


def sample_skill_vector(feature_dim: int, batch_size: int = 1) -> np.ndarray:
    """
    Sample z uniformly from the unit ball in R^feature_dim.
    Method: sample from normal, normalize, scale by uniform radius.
    """
    raw = np.random.randn(batch_size, feature_dim)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    directions = raw / (norms + 1e-8)
    radii = np.random.uniform(0, 1, size=(batch_size, 1)) ** (1.0 / feature_dim)
    return (directions * radii).astype(np.float32)


class HILPFeatureTrainer:
    """
    Trains the HILP feature network using contrastive temporal distance loss.

    Positive pairs: (s_t, s_{t+k}) from same trajectory
    Negative pairs: (s_t, s_random) from different trajectories
    Loss: InfoNCE contrastive loss on squared feature distances.
    """

    def __init__(
        self,
        feature_net: HILPFeatureNetwork,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.feature_net = feature_net.to(device)
        self.optimizer = torch.optim.Adam(feature_net.parameters(), lr=lr)
        self.device = device

    def train_step(
        self,
        obs_anchor: torch.Tensor,
        obs_positive: torch.Tensor,
        obs_negative: torch.Tensor,
    ) -> float:
        self.feature_net.train()
        self.optimizer.zero_grad()

        xi_anchor = self.feature_net(obs_anchor.to(self.device))
        xi_positive = self.feature_net(obs_positive.to(self.device))
        xi_negative = self.feature_net(obs_negative.to(self.device))

        # Squared distances
        d_pos = ((xi_anchor - xi_positive) ** 2).sum(dim=-1)
        d_neg = ((xi_anchor - xi_negative) ** 2).sum(dim=-1)

        # InfoNCE-style contrastive loss
        logits = torch.stack([-d_pos, -d_neg], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        loss = nn.functional.cross_entropy(logits, labels)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path: str):
        torch.save(self.feature_net.state_dict(), path)

    def load(self, path: str):
        self.feature_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
