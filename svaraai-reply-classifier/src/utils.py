import random, numpy as np
def set_seed(seed=42):
    random.seed(seed)
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
