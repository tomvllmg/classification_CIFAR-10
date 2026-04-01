from hydra.utils import instantiate

def build_scheduler(cfg_scheduler, optimizer):
    """
    Entrées : Dictionnaire cfg_scheduler, Optimiseur
    Sortie : Scheduler
    Fonction : Construit le scheduler de taux d'apprentissage.
    """
    if cfg_scheduler is None or cfg_scheduler.get("name") == "none":
        return None
 
    # Hydra instancie la classe, ainsi que les arguments du YAML.
    scheduler = instantiate(cfg_scheduler.param, optimizer=optimizer)
    
    return scheduler
