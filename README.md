# manabot

A reinforcement learning framework for [Magic: The Gathering](https://magic.wizards.com/), built on top of [managym](https://github.com/jacklionheart/managym).

manabot currently uses PPO with a MLP backing a shared value/policy network.

### Training

```bash
# Clone the repo
git clone https://github.com/jacklionheart/manabot.git
cd manabot
# Install the package
pip install -e .
# Run the training script
python manabot/scripts/train.py
```

Edit `manabot/scripts/conf/config.yaml` to configure the experiment.

To see in tensorboard, run: (assumes `quick_train` is the experiment name in `config.yaml`)
```bash
tensorboard tensorboard --logdir ~/manabot-runs/quick_train
```

Current status: training runs without error, but loss is not decreasing. Still very early

### Inference

TBD

### Testing

```bash
pytest tests/
```

### Architecture

manabot is organized into four three components:

1. **`manabot.env`**: 
   - `VectorEnv`: gymnasium.AsyncVectorEnv-based interface around managym
   - `ObservationSpace`: dataclass describing the observation space
   - `Match`: dataclass describing the game of magic to be played (decklists, etc.)
   - `Reward`: dataclass describing the reward function

2. **`manabot.ppo`**: PPO implementation  
   - `Agent`: Shared Value/Policy network
   - `Trainer`: PPO trainer for learning network weights

3. **`manabot.infra`**: Training infrastructure
   - `Experiment`: Experiment tracking with wandb/tensorboard
   - `Hypers`: Hydra-compatible hyperparameter management and configuration


## Architecture

ManaBot is organized into three main components with strict dependency ordering:

1. **`manabot.data`**  
   - Depends only on `managym`.  

2. **`manabot.env`**  
   - Implements a gymnasium-like API around managym.  
   - Depends on `managym`  

3. **`manabot.ppo`**  
   - Contains model architecture and training code.  
   - Depends on `manabot.env` and `manabot.data`.

### Design Decisions

- **managym vs. manabot Split**  
  - **managym** (C++): Handles low-level game logic. Eventually: add native-Cpp inference for faster rollouts
  - **manabot** (Python): Torch-backed optimization, and, for now, inference for rollouts.

- **Dynamic Discrete Action Space**  
  - Limited number of action slots whose meaning can vary significantly from step to step.  
  - The “meaning” of each action slot is included in the observation tensor for context.

## Style Guide

### Code Organization

Files should follow this template:

```python

"""
filename.py
One-line purpose of file

Instructions for collaborators (both human and LLM) on how to approach understanding and editing the code.
Keep this section focused and impactful.
"""

# Standard library
import os
from typing import Dict, List

# Third-party imports
from import torch import Tensor

# manabot imports
from manabot.env import ObservationSpace

# Local imports
from .sibling import Thing
```

### Documentation Principles

- Public APIs should have docstrings focused on clarifying behavior and resolving ambiguities
- Implementation details and design rationale belong in comments or READMEs
- Use organizational comments liberally:

```python
# -----------------------------------------------------------------------------
# Organizational Header
# -----------------------------------------------------------------------------
```

## Code Style

- Follow PEP8
- Use type hints as much as possible
- Use dataclasses where appropriate

When working with this codebase, LLMs should:
- Avoid comments that are transient/denote changes. Imagine the code will be directly copied into the codebase for eternity.
- Pay special attention to file headers and README content for context
- Propose small, iterative changes
- End responses with:
  1. Full implementations of changed files that can be copied into the codebase 
  2. Questions that could clarify intent
  3. Notes on what was intentionally left out

## Future Roadmap

Long-term goals and exploration areas:
- Attentional model architecture
- Unknown information estimation  
- Performance: C++ side inference

