import os 
from tqdm import tqdm

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
from transformers import BertTokenizer, BertModel

from models import * 


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


class BertEncodedSentimentDataset(Dataset):
    def __init__(self, split, data_dir=f'data/bert_sentiment'):
        assert split in ('train', 'val', 'test')
        self.xs = torch.load(os.path.join(data_dir, split, f'{split}_x.pt'))
        self.ys = torch.load(os.path.join(data_dir, split, f'{split}_y.pt'))

        print(f"len xs: {self.xs.shape[0]}")

        assert self.xs.shape[0] == self.ys.shape[0]
    
    def get_info(self):
        return {
            'input_dim': (self.xs.shape[1],),
            'num_classes': self.ys.shape[1]
        }

    def __len__(self):
        return self.xs.shape[0]
    
    def __getitem__(self, idx):
        return self.xs[idx, :], self.ys[idx, :]


def load_bert_encoded_sentiment(batch_size):
    data_dir = f'data/bert_sentiment'
    train_dataloader = DataLoader(BertEncodedSentimentDataset('train', data_dir=data_dir),
                                  batch_size, shuffle=True)
    val_dataloader = DataLoader(BertEncodedSentimentDataset('val', data_dir=data_dir),
                                  batch_size, shuffle=False)
    test_dataloader = DataLoader(BertEncodedSentimentDataset('test', data_dir=data_dir),
                                  batch_size, shuffle=False)
    info = train_dataloader.dataset.get_info()
    return train_dataloader, val_dataloader, test_dataloader, info


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


def create_sentiment():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Device: {device}")

    split_names = ('train', 'val', 'test')
    save_path = f'data/bert_sentiment'
    if not os.path.isdir(save_path):
        for split_name in split_names:
            os.makedirs(os.path.join(save_path, split_name))

    sentiment_data = load_sentiment(8)
    # model = BertEncodedMLP(sentiment_data[-1]).to(device)
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    for split_name, split in zip(('train', 'val', 'test'), sentiment_data[:-1]):
        print(f"Saving {split_name} samples")
        
        xs = []
        ys = []
        
        for batch in tqdm(split):
            # xs.append(model.encoder(batch['input_ids'].to(device)).pooler_output)
            xs.append(model(batch['input_ids'].to(device)).pooler_output)
            # print(xs[0].shape)
            ys.append(batch['labels'].to(device))
        
        xs = torch.concat(xs, dim=0)
        ys = torch.concat(ys, dim=0)

        torch.save(xs, os.path.join(save_path, split_name, f'{split_name}_x.pt'))
        torch.save(ys, os.path.join(save_path, split_name, f'{split_name}_y.pt'))


if __name__ == '__main__':
    create_sentiment()