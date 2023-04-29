import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def load_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST('../data/mnist', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data/mnist', train=False, transform=transform)

    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size

    train_set, val_set = torch.utils.data.random_split(
            train_set, 
            [train_set_size, val_set_size], 
            generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_cifar10(batch_size):
    train_set = datasets.CIFAR10(root='../data/cifar10', download=True, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10(root='../data/cifar10', train=False, transform=transforms.ToTensor())
    
    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size
    
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [train_set_size, val_set_size]
        generator=torch.Generator().manual_seed(1)
    )
    
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader