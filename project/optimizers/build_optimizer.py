from hydra.utils import instantiate

def build_optimizer(cfg_optimizer, model):
    """
    Entrées : Dictionnaire de configuration cfg_optimizer, modèle du réseau
    Sortie : Optimiseur
    Fonction : Construit l'optimiseur en utilisant Hydra avec _target_.
    """
    # Hydra instancie l'optimiseur en lui injectant les paramètres du modèle ainsi que les hyperparamètres de configuration.
    optimizer = instantiate(cfg_optimizer.param, params=model.parameters())
    
    return optimizer
