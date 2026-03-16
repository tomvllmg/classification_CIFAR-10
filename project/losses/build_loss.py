#project/losses/build_loss.py

import torch.nn as nn

def build_loss(cfg_loss):
    """
    Construit la fonction de perte.
    """
    name = cfg_loss.get("name", "cross_entropy").lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
        
    elif name == "mse":
        return nn.MSELoss()
        
    else:
        raise ValueError(f"La fonction de perte '{name}' n'est pas prise en charge.")
