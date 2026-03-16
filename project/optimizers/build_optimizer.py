#project/optimizers/build_optimizer.py

import torch.optim as optim

def build_optimizer(cfg_optimizer, model):
    """
    Construit l'optimiseur PyTorch en fonction de la configuration YAML.
    """
    # On récupère le nom (ex: "adam" ou "sgd") défini dans le YAML
    name = cfg_optimizer.get("name", "adam").lower()
    lr = cfg_optimizer.get("lr", 0.001)
    weight_decay = cfg_optimizer.get("weight_decay", 0.0)

    if name == "adam":
        # On passe les paramètres du modèle à l'optimiseur
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif name == "sgd":
        momentum = cfg_optimizer.get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    else:
        raise ValueError(f"L'optimiseur '{name}' n'est pas pris en charge.")
