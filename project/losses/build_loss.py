from hydra.utils import instantiate

def build_loss(cfg_loss):
    """
    Construit la fonction de perte grâce à l'instanciation dynamique d'Hydra.
    """
    # On instancie directement l'objet pointé par _target_ dans le YAML avec tous les arguments 
    loss_fct = instantiate(cfg_loss.param)
    
    return loss_fct
