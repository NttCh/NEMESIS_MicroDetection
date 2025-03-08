"""
utils.py

Contains utility functions.
"""

import os
import random
import numpy as np
import torch
import datetime

def set_seed(seed: int = 666) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_obj(obj_path: str) -> any:
    """
    Dynamically load an object given its string path.
    
    Example:
      load_obj("torchvision.models.resnet50")
    """
    parts = obj_path.split(".")
    module_path = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = __import__(module_path, fromlist=[obj_name])
    return getattr(module, obj_name)

def thai_time() -> datetime.datetime:
    """
    Returns the current time in Thai time (UTC+7).
    """
    return datetime.datetime.utcnow() + datetime.timedelta(hours=7)
