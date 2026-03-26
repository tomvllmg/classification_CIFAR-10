from hydra.utils import instantiate

def build_scheduler(cfg_scheduler, optimizer):
    """
    Entrées : 
    Sortie :
    Fonction : Construit le scheduler de taux d'apprentissage.
    """
    # Bonne pratique : permettre de ne pas utiliser de scheduler
    if cfg_scheduler is None or cfg_scheduler.get("name") == "none":
        return None

    # Hydra instancie la classe (ex: StepLR) et lui passe l'optimiseur 
    # ainsi que les arguments du YAML (step_size, gamma, etc.)
    scheduler = instantiate(cfg_scheduler.param, optimizer=optimizer)
    
    return scheduler
