import numpy as np 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import skorch
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import BertTokenizer


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

    train_set = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST('data/mnist', train=False, transform=transform)

    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size

    train_set, val_set = torch.utils.data.random_split(
            train_set, 
            [train_set_size, val_set_size], 
            generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    info = {'input_dim': (1, 28, 28), 'num_classes': 10}

    return train_loader, val_loader, test_loader, info


def load_cifar10(batch_size):
    transform = transforms.Compose(
               [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='data/cifar10', download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, transform=transform)
    
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

    info = {'input_dim': (3, 32, 32), 'num_classes': 10}
    
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

    info = {'input_dim': (X_train.shape[0],), 'num_classes': 10}
    
    return train_loader, val_loader, test_loader, info


def load_fashion_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2861,), (0.3530,))
    ])

    train_set = torchvision.datasets.FashionMNIST(root='data/fashion_mnist', download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='data/fashion_mnist', train=False, transform=transform)

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

    info = {'input_dim': (1, 28, 28), 'num_classes': 10}
    
    return train_loader, val_loader, test_loader, info


def load_sentiment(batch_size):
    dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
    labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_data(examples):
        # encode tweets
        text = examples["Tweet"]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        labels_matrix = np.zeros((len(text), len(labels)))
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()

        return encoding
    
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")

    train_dataloader = DataLoader(encoded_dataset["train"], batch_size, shuffle=True)
    val_dataloader = DataLoader(encoded_dataset["validation"], batch_size, shuffle=False)
    test_dataloader = DataLoader(encoded_dataset["test"], batch_size, shuffle=False)

    info = {
        "encoder_dim": (768,),
        "num_classes": 11,
        "tokenizer": tokenizer,
        "id2label": id2label,
        "label2id": label2id,
        "num_classes": 11
    }

    return train_dataloader, val_dataloader, test_dataloader, info