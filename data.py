import torch
import skorch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split


# Over channel dims
def compute_channel_mean_and_var(dataloader, info):
    mean = torch.zeros(info['input_dim'])
    for batch, _ in dataloader:
        mean += torch.mean(batch, dim=0)
    mean /= len(dataloader)
    
    # print(mean.shape)
    assert len(mean.shape) > 1
    mean = torch.mean(mean, dim=[i for i in range(1, len(mean.shape))])
    
    
    stdv = torch.zeros_like(mean)
    mean = mean.reshape(1, -1, *[1 for i in range(len(info['input_dim'])-1)])

    for batch, _ in dataloader:
        stdv += torch.mean((batch - mean)**2, dim=[i for i in range(len(mean.shape)) if i != 1])
    
    stdv = (stdv / len(dataloader)) ** 0.5

    return mean.reshape(stdv.shape), stdv


def load_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('data/mnist', train=False, transform=transform)

    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size

    train_set, val_set = torch.utils.data.random_split(
            train_set, 
            [train_set_size, val_set_size], 
            generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    info = {'input_dim': (1, 28, 28)}

    return train_loader, val_loader, test_loader, info


def load_cifar10(batch_size):
    transform = transforms.Compose(
               [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR10(root='data/cifar10', download=True, transform=transform)
    test_set = datasets.CIFAR10(root='data/cifar10', train=False, transform=transform)
    
    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size
    
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [train_set_size, val_set_size],
        generator=torch.Generator().manual_seed(1)
    )
    
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    info = {'input_dim': (3, 32, 32)}
    
    return train_loader, val_loader, test_loader, info


def load_rcv1(batch_size):
    X, y = fetch_rcv1(data_home='data/rcv1', download_if_missing=True, return_X_y=True)
    # X = X.todense()
    # y = y.todense()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    train_set = skorch.dataset.Dataset(X_train, y_train)
    val_set = skorch.dataset.Dataset(X_val, y_val)
    test_set = skorch.dataset.Dataset(X_test, y_test)
    
    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    info = {'input_dim': (X_train.shape[0],)}
    
    return train_loader, val_loader, test_loader, info


def load_fashion_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2861,), (0.3530,))
    ])

    train_set = datasets.FashionMNIST(root='data/fashion_mnist', download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='data/fashion_mnist', train=False, transform=transform)

    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size
    
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [train_set_size, val_set_size],
        generator=torch.Generator().manual_seed(1)
    )
    
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    info = {'input_dim': (1, 28, 28)}
    
    return train_loader, val_loader, test_loader, info