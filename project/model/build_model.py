
import torch
from .cnn import CNNClassif 

def build_model(cfg_model):
    """
    Construit le modèle CNNClassif à partir de la configuration.
    """
    # On vérifie quand même que le YAML demande bien le bon modèle par sécurité
    name = cfg_model.get("name", "cnn_classif").lower()
    params = cfg_model.get("params", {})

    if name != "cnn_classif":
        raise ValueError(f"Ce projet n'utilise que 'cnn_classif'. Modèle demandé : '{name}'.")

    # On instancie le modèle avec les paramètres du YAML
    model = CNNClassif(**params)

    # --- Initialisation du nn.LazyLinear ---
    # Passage d'un tenseur vide (Batch=1, Canaux=3, 32x32 pour CIFAR-10)
    dummy_input = torch.randn(1, 3, 32, 32)
    model(dummy_input)
    
    return model
