import random
import numpy as np

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from utils import * 
from data import *
from models import * 
from block_sketchy_sgd import * 


BATCH_SIZE = 64
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    train_loader, val_loader, test_loader = load_data(BATCH_SIZE)
    
    model = MLP(28*28, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = BlockSketchySGD(model)
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device).reshape(batch_x.shape[0], -1)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        # loss.backward()
        optimizer.step(loss, model)

        if batch_idx == 0:
            break


if __name__ == '__main__':
    main()