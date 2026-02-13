"""
U2O Pretraining Script for Revolve.

Run once before the Revolve evolutionary loop.
Produces:
  1. hilp_feature_net.pt  - trained HILP feature network
  2. hilp_policy.zip      - pretrained SAC policy (skill-conditioned)
  3. exploration_buffer.npz - collected transitions for bridge step
  4. pretrain_config.json  - config for reproducibility

Usage:
  export ROOT_PATH=/path/to/Revolve
  python -m u2o.pretrain --output_dir ./u2o_pretrained --total_timesteps 500000
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure ROOT_PATH is on sys.path
root_path = os.environ.get("ROOT_PATH", os.path.dirname(os.path.dirname(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from rl_agent.HumanoidEnv import HumanoidEnv
from u2o.networks import HILPFeatureNetwork, HILPFeatureTrainer
from u2o.wrappers import SkillConditionedIntrinsicWrapper


# Dummy reward function for pretraining (returns 0, we use intrinsic reward)
DUMMY_REWARD_FUNC_STR = """
def compute_reward(observation):
    reward = 0.0
    reward_components = {"dummy": 0.0}
    return reward, reward_components
"""


class FeatureTrainingCallback(BaseCallback):
    """
    SB3 callback that trains the HILP feature network periodically
    using transitions from the SAC replay buffer.
    """

    def __init__(
        self,
        feature_trainer: HILPFeatureTrainer,
        obs_dim: int = 376,
        train_freq: int = 1000,
        train_steps: int = 100,
        batch_size: int = 256,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.feature_trainer = feature_trainer
        self.obs_dim = obs_dim
        self.train_freq = train_freq
        self.train_steps = train_steps
        self.batch_size = batch_size
        self._step_count = 0

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.train_freq != 0:
            return True

        replay_buffer = self.model.replay_buffer
        if replay_buffer.size() < self.batch_size * 2:
            return True

        for _ in range(self.train_steps):
            indices = np.random.randint(
                0, replay_buffer.size(), size=self.batch_size
            )
            neg_indices = np.random.randint(
                0, replay_buffer.size(), size=self.batch_size
            )

            # SB3 replay buffer stores obs as (N, 1, obs_dim) for VecEnv
            # Augmented obs is (obs_dim + feature_dim), slice to get raw obs
            obs_anchor = replay_buffer.observations[indices].squeeze(1)[
                :, : self.obs_dim
            ]
            obs_positive = replay_buffer.next_observations[indices].squeeze(1)[
                :, : self.obs_dim
            ]
            obs_negative = replay_buffer.observations[neg_indices].squeeze(1)[
                :, : self.obs_dim
            ]

            loss = self.feature_trainer.train_step(
                torch.FloatTensor(obs_anchor),
                torch.FloatTensor(obs_positive),
                torch.FloatTensor(obs_negative),
            )

        if self.verbose:
            print(
                f"[HILP Feature Training] Step {self._step_count}, "
                f"Loss: {loss:.4f}"
            )

        return True


def create_dummy_humanoid_env() -> HumanoidEnv:
    """Create a HumanoidEnv with a dummy reward function for pretraining."""
    tmp_dir = tempfile.mkdtemp()
    return HumanoidEnv(
        reward_func_str=DUMMY_REWARD_FUNC_STR,
        counter=0,
        generation_id=0,
        island_id="pretrain",
        reward_history_file=os.path.join(tmp_dir, "reward_history.json"),
        velocity_file=os.path.join(tmp_dir, "velocity.txt"),
        model_checkpoint_file=os.path.join(tmp_dir, "checkpoint"),
    )


def pretrain(
    output_dir: str,
    feature_dim: int = 32,
    total_timesteps: int = 500000,
    feature_hidden_dims: tuple = (256, 256),
    feature_lr: float = 3e-4,
    sac_lr: float = 3e-4,
    feature_train_freq: int = 1000,
    feature_train_steps: int = 100,
    intrinsic_reward_scale: float = 1.0,
    device: str = None,
):
    """
    Complete U2O pretraining pipeline.

    Args:
        output_dir: Directory to save pretrained artifacts
        feature_dim: Dimension of HILP feature space (and z)
        total_timesteps: Total timesteps for pretraining
        feature_hidden_dims: Hidden layer sizes for feature network
        feature_lr: Learning rate for feature network
        sac_lr: Learning rate for SAC policy
        feature_train_freq: Train features every N steps
        feature_train_steps: Gradient steps per feature training round
        intrinsic_reward_scale: Scale factor for intrinsic reward
        device: 'cuda' or 'cpu'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    obs_dim = 376

    # 1. Create feature network
    feature_net = HILPFeatureNetwork(
        obs_dim=obs_dim,
        feature_dim=feature_dim,
        hidden_dims=feature_hidden_dims,
    )
    feature_trainer = HILPFeatureTrainer(feature_net, lr=feature_lr, device=device)

    # 2. Create environment with intrinsic reward wrapper
    base_env = create_dummy_humanoid_env()
    wrapped_env = SkillConditionedIntrinsicWrapper(
        env=base_env,
        feature_net=feature_net,
        feature_dim=feature_dim,
        device=device,
        intrinsic_reward_scale=intrinsic_reward_scale,
    )

    # Wrap for SB3
    monitored_env = Monitor(wrapped_env)
    vec_env = DummyVecEnv([lambda: monitored_env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

    # 3. Create SAC model (obs_dim = 376 + feature_dim = 408)
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        learning_rate=sac_lr,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
    )

    # 4. Create feature training callback
    feature_callback = FeatureTrainingCallback(
        feature_trainer=feature_trainer,
        obs_dim=obs_dim,
        train_freq=feature_train_freq,
        train_steps=feature_train_steps,
        verbose=1,
    )

    # 5. Train
    print(f"Starting U2O pretraining for {total_timesteps} timesteps...")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Obs dim: {obs_dim} -> Augmented: {obs_dim + feature_dim}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[feature_callback],
    )

    # 6. Save artifacts
    feature_net_path = os.path.join(output_dir, "hilp_feature_net.pt")
    policy_path = os.path.join(output_dir, "hilp_policy")
    buffer_path = os.path.join(output_dir, "exploration_buffer.npz")
    config_path = os.path.join(output_dir, "pretrain_config.json")

    # Save feature network
    feature_trainer.save(feature_net_path)
    print(f"Saved feature network to {feature_net_path}")

    # Save SAC policy
    model.save(policy_path)
    print(f"Saved pretrained policy to {policy_path}.zip")

    # Save exploration buffer from wrapper
    buffer_data = wrapped_env.get_buffer_arrays()
    if buffer_data is not None:
        np.savez_compressed(buffer_path, **buffer_data)
        print(
            f"Saved exploration buffer ({buffer_data['obs'].shape[0]} transitions) "
            f"to {buffer_path}"
        )
    else:
        print("Warning: No transitions collected for exploration buffer.")

    # Save config for reproducibility
    config = {
        "feature_dim": feature_dim,
        "obs_dim": obs_dim,
        "total_timesteps": total_timesteps,
        "feature_hidden_dims": list(feature_hidden_dims),
        "augmented_obs_dim": obs_dim + feature_dim,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    print("Pretraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U2O Pretraining for Revolve")
    parser.add_argument("--output_dir", type=str, default="./u2o_pretrained")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--feature_lr", type=float, default=3e-4)
    parser.add_argument("--sac_lr", type=float, default=3e-4)
    parser.add_argument("--feature_train_freq", type=int, default=1000)
    parser.add_argument("--feature_train_steps", type=int, default=100)
    parser.add_argument("--intrinsic_reward_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    pretrain(
        output_dir=args.output_dir,
        feature_dim=args.feature_dim,
        total_timesteps=args.total_timesteps,
        feature_lr=args.feature_lr,
        sac_lr=args.sac_lr,
        feature_train_freq=args.feature_train_freq,
        feature_train_steps=args.feature_train_steps,
        intrinsic_reward_scale=args.intrinsic_reward_scale,
        device=args.device,
    )
