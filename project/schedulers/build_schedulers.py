#project/schedulers/build_schedulers.py

import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler(cfg_scheduler, optimizer):
    """
    Construit le planificateur de taux d'apprentissage.
    """
    name = cfg_scheduler.get("name", "cosine").lower()

    if name == "cosine":
        # T_max correspond souvent au nombre total d'époques (epochs)
        t_max = cfg_scheduler.get("T_max", 100) 
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        
    elif name == "step":
        step_size = cfg_scheduler.get("step_size", 30)
        gamma = cfg_scheduler.get("gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    else:
        # Il est courant de ne pas avoir de scheduler, on peut renvoyer None
        if name == "none":
            return None
        raise ValueError(f"Le scheduler '{name}' n'est pas pris en charge.")
