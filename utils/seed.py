import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy, random, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
