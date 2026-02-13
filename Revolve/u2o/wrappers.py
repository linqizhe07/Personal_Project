"""
Gymnasium wrappers for U2O integration.

SkillConditionedWrapper: Augments obs with skill vector z.
  - Pretraining: z sampled per episode from unit ball.
  - Fine-tuning: z fixed to z* from bridge step.

SkillConditionedIntrinsicWrapper: Pretraining only.
  - Augments obs with z AND replaces reward with HILP intrinsic reward.
  - Stores raw transitions for exploration buffer.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
import torch

from u2o.networks import (
    HILPFeatureNetwork,
    compute_intrinsic_reward,
    sample_skill_vector,
)


class SkillConditionedWrapper(gym.Wrapper):
    """
    Wraps an environment to concatenate a skill vector z to observations.

    Original obs: (obs_dim,)  e.g. (376,)
    Augmented obs: (obs_dim + feature_dim,)  e.g. (376 + 32 = 408,)

    During pretraining: z is resampled each episode from unit ball.
    During fine-tuning: z is fixed to z* from the bridge step.
    """

    def __init__(
        self,
        env: gym.Env,
        feature_dim: int = 32,
        fixed_z: Optional[np.ndarray] = None,
    ):
        super().__init__(env)
        self.feature_dim = feature_dim
        self.fixed_z = fixed_z
        self.current_z = None

        # Modify observation space
        orig_obs_space = env.observation_space
        low = np.concatenate(
            [orig_obs_space.low, -np.ones(feature_dim) * np.inf]
        )
        high = np.concatenate(
            [orig_obs_space.high, np.ones(feature_dim) * np.inf]
        )
        self.observation_space = spaces.Box(
            low=low.astype(np.float64),
            high=high.astype(np.float64),
            dtype=np.float64,
        )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result

        if self.fixed_z is not None:
            self.current_z = self.fixed_z.copy()
        else:
            self.current_z = sample_skill_vector(self.feature_dim, batch_size=1)[0]

        return np.concatenate([obs, self.current_z])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        augmented_obs = np.concatenate([obs, self.current_z])
        return augmented_obs, reward, terminated, truncated, info


class SkillConditionedIntrinsicWrapper(gym.Wrapper):
    """
    Combined wrapper for pretraining:
      - Augments obs with z (sampled per episode)
      - Replaces env reward with HILP intrinsic reward
      - Stores raw transitions for exploration buffer

    Usage:
        env = HumanoidEnv(...)
        env = SkillConditionedIntrinsicWrapper(env, feature_net, feature_dim)
    """

    def __init__(
        self,
        env: gym.Env,
        feature_net: HILPFeatureNetwork,
        feature_dim: int = 32,
        device: str = "cpu",
        intrinsic_reward_scale: float = 1.0,
    ):
        super().__init__(env)
        self.feature_net = feature_net
        self.feature_dim = feature_dim
        self.device = device
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.current_z = None
        self.prev_obs = None

        # Buffer storage for exploration_buffer.npz
        self._buffer_obs = []
        self._buffer_next_obs = []
        self._buffer_actions = []
        self._buffer_extrinsic_rewards = []

        # Modify observation space: original + z
        orig = env.observation_space
        low = np.concatenate([orig.low, -np.ones(feature_dim) * np.inf])
        high = np.concatenate([orig.high, np.ones(feature_dim) * np.inf])
        self.observation_space = spaces.Box(
            low=low.astype(np.float64),
            high=high.astype(np.float64),
            dtype=np.float64,
        )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs = result[0] if isinstance(result[0], np.ndarray) else result
        else:
            obs = result

        self.current_z = sample_skill_vector(self.feature_dim, batch_size=1)[0]
        self.prev_obs = obs.copy()

        return np.concatenate([obs, self.current_z])

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Compute intrinsic reward
        with torch.no_grad():
            prev_t = torch.FloatTensor(self.prev_obs).unsqueeze(0).to(self.device)
            curr_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            z_t = torch.FloatTensor(self.current_z).unsqueeze(0).to(self.device)
            xi_prev = self.feature_net(prev_t)
            xi_curr = self.feature_net(curr_t)
            intrinsic_r = compute_intrinsic_reward(xi_prev, xi_curr, z_t)
            reward = intrinsic_r.item() * self.intrinsic_reward_scale

        # Store raw transition for bridge step buffer
        self._buffer_obs.append(self.prev_obs.copy())
        self._buffer_next_obs.append(obs.copy())
        self._buffer_actions.append(action.copy())
        self._buffer_extrinsic_rewards.append(env_reward)

        self.prev_obs = obs.copy()
        info["extrinsic_reward"] = env_reward

        augmented_obs = np.concatenate([obs, self.current_z])
        return augmented_obs, reward, terminated, truncated, info

    def get_buffer_arrays(self):
        """Export stored transitions as numpy arrays for bridge step."""
        if len(self._buffer_obs) == 0:
            return None
        return {
            "obs": np.array(self._buffer_obs),
            "next_obs": np.array(self._buffer_next_obs),
            "actions": np.array(self._buffer_actions),
            "extrinsic_rewards": np.array(self._buffer_extrinsic_rewards),
        }
