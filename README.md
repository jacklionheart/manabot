# ManaBot

A reinforcement learning framework for Magic: The Gathering.

## Installation

```zsh
git clone https://github.com/jacklionheart/manabot.git
cd manabot
pip install -e .
```

## Testing

```zsh
pip install -e .
pytest tests/
```


## Style Guide


### Import Conventions

Imports should be organized in the following groups, separated by blank lines:

1. Standard library imports
2. Third-party library imports
3. Global imports from the manabot package (absolute)
4. Local imports from sibling files (relative)

Default towards "from x import y" imports.


```python
# Standard library
import json
from typing import Dict, List, Optional

# Third-party
import numpy as np
import torch
from torch import nn

# Local imports
from .env import Env                           # Same directory - use dot
from manabot.data.observation import Observation   # Different directory - absolute
