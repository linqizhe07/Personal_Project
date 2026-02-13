# REvolve + U2O: Reward Evolution with Unsupervised-to-Online RL
******************************************************
**Based on the official ICLR 2025 REvolve paper, extended with U2O (Unsupervised-to-Online RL) pretraining.**


## Overview

REvolve uses LLM-guided evolutionary algorithms to automatically design reward functions for RL. This fork integrates **U2O (Unsupervised-to-Online RL)** to address the "blind search" problem: instead of training each candidate reward function from scratch, we pretrain a skill-conditioned policy via HILP and fine-tune it per candidate, yielding faster evaluation and more stable fitness signals.

```
[Pretrain (once)]                        [Evolution Loop (per candidate)]

HumanoidEnv + intrinsic reward           LLM generates reward function
        |                                        |
HILP feature network xi(s)               Bridge: z* = argmin ||r - F^T z||
        |                                        |
Skill policy pi(a|s,z)                   Fine-tune pretrained policy with z*
        |                                        |
exploration_buffer.npz                   Evaluate fitness (velocity)
```

## Setup
```shell
git clone https://github.com/RishiHazra/Revolve.git
cd Revolve
conda create -n "revolve" python=3.10
conda activate revolve
pip install -e .
```

## Run (Original REvolve)

```shell
export ROOT_PATH='Revolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=15 \
        database.num_islands=5 \
        database.max_island_size=8 \
        data_paths.run=10 \
        environment.name="HumanoidEnv"
```

## Run (REvolve + U2O)

### Step 1: Pretrain (one-time)

Train a HILP skill-conditioned policy with unsupervised intrinsic rewards. This produces three artifacts: feature network, pretrained SAC policy, and an exploration buffer.

```shell
export ROOT_PATH='Revolve'
python -m u2o.pretrain \
        --output_dir ./u2o_pretrained \
        --feature_dim 32 \
        --total_timesteps 500000
```

| Output File | Description |
|-------------|-------------|
| `hilp_feature_net.pt` | HILP temporal distance feature network xi(s) |
| `hilp_policy.zip` | Pretrained SAC policy (obs_dim=408: 376 + 32) |
| `exploration_buffer.npz` | Collected transitions for bridge step |
| `pretrain_config.json` | Config for reproducibility |

### Step 2: Run Evolution with U2O

```shell
export ROOT_PATH='Revolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        u2o.enabled=true \
        u2o.pretrained_dir=./u2o_pretrained \
        u2o.feature_dim=32 \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=15 \
        database.num_islands=5 \
        database.max_island_size=8 \
        data_paths.run=10 \
        environment.name="HumanoidEnv"
```

When `u2o.enabled=true`, each candidate reward function goes through:
1. **Bridge**: Linear regression finds optimal skill vector z* for the candidate reward
2. **Fine-tune**: Loads pretrained policy, fixes z=z*, trains with LLM-generated reward

When `u2o.enabled=false` (default), the original REvolve pipeline runs unchanged.

## Project Structure

```
Revolve/
├── main.py                     # Main evolutionary loop
├── modules.py                  # Reward generation, policy training, evaluation
├── rewards_database.py         # Island-based population management
├── utils.py                    # Utilities
├── prompts/                    # LLM prompts (mutation, crossover)
├── evolutionary_utils/         # Island/Individual entities
├── rl_agent/                   # RL training and evaluation
│   ├── main.py                 # SAC training (+ U2O fine-tune functions)
│   ├── HumanoidEnv.py          # Humanoid environment
│   ├── AdroitEnv.py            # Adroit manipulation environment
│   └── evaluate.py             # Fitness evaluation
├── u2o/                        # U2O integration (new)
│   ├── networks.py             # HILP feature network + trainer
│   ├── wrappers.py             # Skill-conditioned gym wrappers
│   ├── bridge.py               # Bridge step (z* via linear regression)
│   └── pretrain.py             # Pretraining script
├── human_feedback/             # Elo scoring for human evaluation
└── cfg/                        # Hydra configs
    ├── generate.yaml           # Evolution + U2O config
    └── train.yaml              # RL training config
```

## U2O Configuration

All U2O parameters in `cfg/generate.yaml`:

```yaml
u2o:
  enabled: false                  # toggle U2O on/off
  pretrained_dir: ./u2o_pretrained  # path to pretrained artifacts
  feature_dim: 32                 # HILP feature / skill vector dimension
  bridge:
    regularization: 1e-4          # L2 regularization for z* regression
```

Pretrain script arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `./u2o_pretrained` | Output directory |
| `--feature_dim` | `32` | HILP feature dimension |
| `--total_timesteps` | `500000` | Pretraining timesteps |
| `--feature_lr` | `3e-4` | Feature network learning rate |
| `--sac_lr` | `3e-4` | SAC policy learning rate |
| `--feature_train_freq` | `1000` | Feature training frequency (steps) |
| `--feature_train_steps` | `100` | Gradient steps per feature training |
| `--intrinsic_reward_scale` | `1.0` | Intrinsic reward scale |
| `--device` | `auto` | `cuda` or `cpu` |

## Other Utilities
* The prompts are listed in `prompts/` folder.
* Elo scoring in `human_feedback/` folder.

*Note, we will soon release the AirSim environment setup script.*

For AirSim, follow the instruction on this link [https://microsoft.github.io/AirSim/build_linux/](AirSim)
```shell
export AIRSIM_PATH='AirSim'
export AIRSIMNH_PATH='AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh'
```

## Citation

### To cite the original REvolve paper:
```bibtex
@inproceedings{hazra2025revolve,
	title        = {{RE}volve: Reward Evolution with Large Language Models using Human Feedback},
	author       = {Rishi Hazra and Alkis Sygkounas and Andreas Persson and Amy Loutfi and Pedro Zuidberg Dos Martires},
	year         = 2025,
	booktitle    = {The Thirteenth International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=cJPUpL8mOw}
}
```

### U2O reference:
```bibtex
@inproceedings{lee2024unsupervised,
	title        = {Unsupervised-to-Online Reinforcement Learning},
	author       = {Junsu Lee and Seohong Park and Sergey Levine},
	year         = 2024,
	url          = {https://arxiv.org/abs/2408.14785}
}
```
