import hydra
from omegaconf import DictConfig
import torch

# Importation des modules internes
from project.model.cnn import CNNClassif
from project.data.dataset import get_dataloaders # Fonction fictive pour charger CIFAR-10
from project.losses.build_loss import build_loss
from project.optimizers.build_optimizer import build_optimizer
from project.schedulers.build_schedulers import build_scheduler
from project.utils.training_loop import train_model

@hydra.main(
  version_base=None,
  config_path="project/config",
  config_name="config"
)
def main(cfg: DictConfig):
  """
  Fonction principale exécutée lors du lancement de train.py
  'cfg' contient toute la configuration assemblée depuis vos fichiers YAML.
  """
   # 1. Configuration du matériel (Utiliser la carte graphique si possible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement lancé sur : {device}")

    # 2. Préparation des données
    # On passe la partie "data" de la configuration (taille des batchs, etc.)
    train_loader, val_loader = get_dataloaders(cfg.data)

    # 3. Création du modèle et envoi sur le bon appareil (CPU ou GPU)
    model = CNNClassif(cfg.model).to(device)

    # 4. Assemblage via vos "Builders"
    # On utilise les fonctions que nous avons vues précédemment
    criterion = build_loss(cfg.loss)
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = build_scheduler(cfg.scheduler, optimizer)

    # 5. Lancement du moteur d'entraînement
    print("Début de la boucle d'entraînement...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.epochs, # Défini dans votre config.yaml principal
        device=device
    )
    
    print("Entraînement terminé !")


if __name__ == "__main__":
  main()
