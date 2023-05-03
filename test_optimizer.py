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
    train_loader, val_loader, test_loader = load_mnist(BATCH_SIZE)
    
    model = MLP(28*28, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = BlockSketchySGD(model)
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device).reshape(batch_x.shape[0], -1)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        print(loss)

        optimizer.zero_grad()
        # loss.backward()
        optimizer.step(loss, model)

        # if batch_idx == 0:
        #     break


# This is a test to figure out what is going on in the method to compute HVPs that does 
# not return null. It seems like based on the output it is actually computing H.T @ v 
# instead of H @ v.
# EDIT: It is computing H.T, verified from 
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
def hvp_test():
    p = torch.tensor([ [1., 1.], [0., 1.] ], requires_grad=True)
    A = torch.tensor([ [2., 4.], [3., 2.] ], requires_grad=True)
    g = A @ p 

    J = torch.tensor([ [2, 0, 4, 0], [0, 2, 0, 4], [3, 0, 2, 0], [0, 3, 0, 2] ], dtype=torch.float32, requires_grad=False).T

    # v = torch.randn_like(p)
    # v = torch.ones_like(p)
    v = torch.tensor([[1., 0.], [0., 0.]])
    y = J @ v.reshape(-1,)

    jvp = torch.autograd.grad(
        # g.view(-1), p.view(-1), 
        g, p,
        grad_outputs=v,
        only_inputs=True, allow_unused=True, 
        retain_graph=True
    )

    print(f"{y}")
    print(f"{jvp}")


if __name__ == '__main__':
    # hvp_test()
    main()