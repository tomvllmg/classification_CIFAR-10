import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(cfg_data):
    """
    Prépare et retourne les DataLoaders pour l'entraînement et la validation sur CIFAR-10.
    """
    # 1. Récupération des paramètres depuis votre configuration (ex: config.yaml)
    data_dir = cfg_data.get("data_dir", "./data")
    batch_size = cfg_data.get("batch_size", 128)
    num_workers = cfg_data.get("num_workers", 2) # Nombre de cœurs CPU alloués au chargement
    
    # 2. Définition des transformations (Prétraitement et Augmentation)
    # Les valeurs de normalisation ci-dessous sont les standards mathématiques pour CIFAR-10
    
    # Transformations pour l'entraînement (avec augmentation de données)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Ajoute du padding puis recadre aléatoirement
        transforms.RandomHorizontalFlip(),    # Retourne l'image horizontalement (effet miroir)
        transforms.ToTensor(),                # Convertit l'image en Tenseur PyTorch (valeurs entre 0 et 1)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalisation
    ])

    # Transformations pour la validation/test (SANS augmentation de données, juste le strict nécessaire)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 3. Téléchargement et création des Datasets
    # Si les données ne sont pas dans 'data_dir', PyTorch les télécharge (download=True)
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # 4. Création des DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,       # On mélange les données à chaque époque (très important pour l'entraînement !)
        num_workers=num_workers,
        pin_memory=True     # Optimisation pour accélérer le transfert des données vers la carte graphique (GPU)
    )
    
    val_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,      # Pas besoin de mélanger pour la validation
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
