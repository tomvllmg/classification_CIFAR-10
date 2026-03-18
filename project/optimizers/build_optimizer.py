# project/optimizers/build_optimizer.py

from hydra.utils import instantiate

def build_optimizer(cfg_optimizer, model):
    """
    Construit l'optimiseur en utilisant la magie de Hydra (_target_).
    """
    # Hydra lit `_target_: torch.optim.Adam` dans ton YAML 
    # et crée l'optimiseur en lui passant les paramètres du modèle !
    optimizer = instantiate(cfg_optimizer.param, params=model.parameters())
    
    return optimizer
