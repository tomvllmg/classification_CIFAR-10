import hydra
from omegaconf import DictConfig
import torch

# Importation de ta fonction pour charger les données
from data.dataloader import build_dataloaders

# ==============================================================
# 🛠️ PATCH POUR LA COMPATIBILITÉ HYDRA / PYTHON 3.14
import argparse
orig_expand_help = argparse.HelpFormatter._expand_help
def patched_expand_help(self, action):
    if action.help is not None and type(action.help).__name__ == 'LazyCompletionHelp':
        action.help = str(action.help)
    return orig_expand_help(self, action)
argparse.HelpFormatter._expand_help = patched_expand_help
# ==============================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "="*50)
    print("🧪 ÉVALUATION DU MEILLEUR MODÈLE SUR LE JEU DE TEST")
    print("="*50)

    # 1. Définition du device (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution sur : {device}")

    # 2. Chargement UNIQUEMENT du jeu de test
    # On met "_" pour ignorer le train et le val_loader
    _, _, test_loader = build_dataloaders(cfg.data, cfg.augmentation)
    print(f"Jeu de test chargé : {len(test_loader.dataset)} images prêtes.")

    # 3. Chargement du modèle champion sauvegardé par Optuna
    try:
        model = torch.load('best_optuna_model.pt', map_location=device, weights_only=False)
        model.eval() # On passe le modèle en mode évaluation (désactive le dropout, etc.)
        print("✅ Modèle 'best_optuna_model.pt' chargé avec succès !")
    except FileNotFoundError:
        print("❌ Erreur : Le fichier 'best_optuna_model.pt' est introuvable.")
        print("As-tu bien laissé Optuna terminer au moins un essai ?")
        return

    # 4. Boucle d'évaluation sur le Test Set
    correct = 0
    total = 0

    # On désactive le calcul du gradient car on n'entraîne plus le modèle
    with torch.no_grad():
        for images, labels in test_loader:
            # On envoie les données sur le bon device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)

            # Prédictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 5. Affichage du score final
    accuracy = 100 * correct / total
    print("\n" + "🏆 "*10)
    print(f"SCORE FINAL SUR LE JEU DE TEST : {accuracy:.2f}%")
    print("🏆 "*10 + "\n")

if __name__ == "__main__":
    main()