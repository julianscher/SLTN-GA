import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def get_project_root_path():
    # Walk upward until we find a .git folder or other known marker
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Repository root not found")

def get_results_path(results_path=None):
    root = get_project_root_path()
    return root / "study_out" if results_path is None else Path(results_path)

def get_resources_path():
    return get_project_root_path() / "resources"

def get_configs_path():
    return get_project_root_path() / "configs"


def set_seed(seed=None, only_model=False):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not only_model:
            np.random.seed(seed)
            random.seed(seed)