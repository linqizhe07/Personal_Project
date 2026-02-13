"""
U2O Bridge Step: Find optimal skill vector z* for a given task reward.

Given:
  - Pretrained feature network xi(s)
  - Exploration buffer of transitions {(s, s', a, r_env)}
  - A candidate reward function compute_reward()

The bridge step:
  1. Evaluate candidate reward on all buffer transitions -> r_task[i]
  2. Compute features f[i] = xi(s'[i]) - xi(s[i]) for all transitions
  3. Solve z* = argmin_z ||r_task - F z||^2 via linear regression
  4. Compute reward scale alpha = std(r_task) / std(F z*)
  5. Return z* and alpha
"""

import inspect
import numpy as np
import torch
from typing import Tuple, Callable, Dict

from u2o.networks import HILPFeatureNetwork


def build_env_state_from_obs(obs: np.ndarray) -> Dict:
    """
    Converts a raw observation array into the env_state dict
    that reward functions expect.
    Matches rl_agent/environment.py CustomEnvironment.env_state.
    """
    return {"observation": obs}


def evaluate_reward_on_buffer(
    reward_func: Callable,
    buffer: Dict[str, np.ndarray],
    env_state_builder: Callable = build_env_state_from_obs,
) -> np.ndarray:
    """
    Evaluate a candidate reward function on all transitions in the buffer.

    Args:
        reward_func: The LLM-generated compute_reward function
        buffer: dict with keys 'obs', 'next_obs', shapes (N, obs_dim)
        env_state_builder: converts obs array -> env_state dict

    Returns:
        rewards: (N,) array of task rewards
    """
    N = buffer["obs"].shape[0]
    rewards = np.zeros(N)

    params = inspect.signature(reward_func).parameters

    for i in range(N):
        obs = buffer["next_obs"][i]
        env_state = env_state_builder(obs)
        args_to_pass = {
            param: env_state[param] for param in params if param in env_state
        }
        try:
            reward, _ = reward_func(**args_to_pass)
            rewards[i] = float(reward)
        except Exception:
            rewards[i] = 0.0

    return rewards


def compute_features_on_buffer(
    feature_net: HILPFeatureNetwork,
    buffer: Dict[str, np.ndarray],
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Compute f[i] = xi(s'[i]) - xi(s[i]) for all transitions in buffer.

    Args:
        feature_net: Pretrained HILP feature network
        buffer: dict with 'obs' and 'next_obs', shapes (N, obs_dim)
        device: torch device
        batch_size: batch size for inference

    Returns:
        features: (N, feature_dim) array of delta-xi features
    """
    feature_net.eval()
    N = buffer["obs"].shape[0]
    all_features = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            obs_batch = torch.FloatTensor(buffer["obs"][start:end]).to(device)
            next_obs_batch = torch.FloatTensor(
                buffer["next_obs"][start:end]
            ).to(device)

            xi_s = feature_net(obs_batch)
            xi_s_next = feature_net(next_obs_batch)
            delta_xi = (xi_s_next - xi_s).cpu().numpy()
            all_features.append(delta_xi)

    return np.concatenate(all_features, axis=0)


def bridge_step(
    reward_func: Callable,
    feature_net: HILPFeatureNetwork,
    buffer: Dict[str, np.ndarray],
    env_state_builder: Callable = build_env_state_from_obs,
    device: str = "cpu",
    regularization: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """
    Complete bridge step: find z* and reward scale alpha.

    Args:
        reward_func: LLM-generated candidate reward function
        feature_net: Pretrained HILP feature network
        buffer: Exploration buffer with 'obs', 'next_obs' arrays
        env_state_builder: converts obs -> env_state dict
        device: torch device
        regularization: L2 regularization for linear regression

    Returns:
        z_star: (feature_dim,) optimal skill vector
        alpha: reward scale factor
    """
    # Step 1: Evaluate task reward on buffer
    r_task = evaluate_reward_on_buffer(reward_func, buffer, env_state_builder)

    # Step 2: Compute features
    F = compute_features_on_buffer(feature_net, buffer, device)

    # Step 3: Linear regression z* = (F^T F + lambda I)^{-1} F^T r_task
    FtF = F.T @ F + regularization * np.eye(F.shape[1])
    Ftr = F.T @ r_task
    z_star = np.linalg.solve(FtF, Ftr)

    # Normalize z_star to unit ball (project if norm > 1)
    z_norm = np.linalg.norm(z_star)
    if z_norm > 1.0:
        z_star = z_star / z_norm

    # Step 4: Compute reward scale
    predicted = F @ z_star
    std_task = np.std(r_task) + 1e-8
    std_pred = np.std(predicted) + 1e-8
    alpha = std_task / std_pred

    return z_star.astype(np.float32), float(alpha)
