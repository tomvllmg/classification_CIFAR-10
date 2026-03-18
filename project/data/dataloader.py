import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def build_dataloaders(cfg_data, cfg_aug):
    # Tes valeurs exactes de normalisation !
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    # Gestion de l'augmentation
    if cfg_aug.get("name") == "basic":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = val_transform # Pas d'augmentation

    # 1. Chargement des données brutes
    dataset_train_aug = torchvision.datasets.CIFAR10(root=cfg_data.data_dir, train=True, download=True, transform=train_transform)
    dataset_val_clean = torchvision.datasets.CIFAR10(root=cfg_data.data_dir, train=True, download=True, transform=val_transform)
    dataset_test = torchvision.datasets.CIFAR10(root=cfg_data.data_dir, train=False, download=True, transform=val_transform)

    # 2. Le Split 80/20
    num_train = len(dataset_train_aug)
    split = int(0.8 * num_train)
    
    torch.manual_seed(42)
    indices = torch.randperm(num_train).tolist()
    train_idx, valid_idx = indices[:split], indices[split:]
    
    # ==========================================
    # L'ASTUCE POUR LE CPU (Fast Dev Run)
    # ==========================================
    if cfg_data.get("debug_mode", False):
        print("⚠️ DEBUG MODE ACTIF : Utilisation d'un sous-échantillon des données (CPU Friendly)")
        # On ne garde que les 800 premiers du train, 200 du valid
        train_idx = train_idx[:800]
        valid_idx = valid_idx[:200]
        
        # Et on bride le test set à 100
        test_indices = torch.arange(100).tolist()
        dataset_test = Subset(dataset_test, test_indices)

    # 3. Création des subsets pour Train et Valid
    train_subset = Subset(dataset_train_aug, train_idx)
    valid_subset = Subset(dataset_val_clean, valid_idx)

    # 4. DataLoaders
    train_loader = DataLoader(train_subset, batch_size=cfg_data.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=cfg_data.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=cfg_data.batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
