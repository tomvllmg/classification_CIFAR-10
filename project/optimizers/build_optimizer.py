from hydra.utils import instantiate

def build_optimizer(cfg_optimizer, model):
    """
    Entrées : Dictionnaire de configuration cfg_optimizer, modèle du réseau
    Sortie : Optimiseur
    Fonction : Construit l'optimiseur en utilisant Hydra avec _target_.
    """
    # Hydra exploite le champ _target_ du fichier YAML pour instancier dynamiquement l'optimiseur (ici torch.optim.Adam)
    # en lui injectant les paramètres du modèle ainsi que les hyperparamètres de configuration.
    optimizer = instantiate(cfg_optimizer.param, params=model.parameters())
    
    return optimizer
