from tqdm import tqdm
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


OPTIMIZERS = {'Adam': optim.Adam, 'BlockSketchySGD': BlockSketchySGD}


BATCH_SIZE = 64
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    num_correct = 0
    num_total = len(val_loader.dataset)

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device).reshape(batch_x.shape[0], -1)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            val_loss += criterion(logits, batch_y).item() * batch_x.shape[0]
            num_correct += (logits.argmax(1) == batch_y).sum().item()

    val_loss /= len(val_loader)*BATCH_SIZE
    val_accuracy = num_correct / num_total 

    return val_loss, val_accuracy


def train(model, train_loader, val_loader, opt_name, num_epochs=2, lr=3e-4, verbose=False):
    criterion = nn.CrossEntropyLoss()
    if opt_name != 'BlockSketchySGD':
        optimizer = OPTIMIZERS[opt_name](model.parameters(), lr=lr)
    else:
        optimizer = OPTIMIZERS[opt_name](model, lr=lr)

    running_loss = 0.0      # Avg loss over the past 100 samples

    out_timesteps = []
    out_train_loss = []
    out_val_loss = []
    out_val_acc = []
    
    for epoch_idx in range(num_epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device).reshape(batch_x.shape[0], -1)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            if opt_name != 'BlockSketchySGD':
                loss.backward()
                optimizer.step()
            else:
                optimizer.step(loss)

            running_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                val_loss, val_acc = evaluate(model, val_loader)
                if verbose:
                    out_str = f'Epoch {epoch_idx} Batch num {batch_idx}: '
                    out_str = out_str + f'train_loss={running_loss / 100:.3f}, val_loss={val_loss:.3f}, '
                    out_str = out_str + f'val_acc={val_acc:.3f}'
                    print(out_str)

                out_timesteps.append(len(train_loader) * BATCH_SIZE * epoch_idx + BATCH_SIZE * batch_idx)
                out_train_loss.append(running_loss / 100)
                out_val_loss.append(val_loss)
                out_val_acc.append(val_acc)

                running_loss = 0.0

    return out_timesteps, out_train_loss, out_val_loss, out_val_acc


def run_experiment():
    train_loader, val_loader, test_loader = load_mnist(BATCH_SIZE)

    for opt_name in OPTIMIZERS:
        seed(0)
        model = MLP(28*28, 10).to(device)
        
        train_logs = \
            train(model, train_loader, val_loader, num_epochs=3, opt_name=opt_name, lr=LR, verbose=True)
        final_val_perf = evaluate(model, val_loader)
    
        vizualize_results(opt_name, *train_logs)
    
    plt.legend()
    plt.show()


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    run_experiment()

if __name__ == '__main__':
    main()


# Some random stuff I was trying
# def train_experimental():
#     criterion = nn.CrossEntropyLoss()

#     params = {k: v.detach() for k, v in model.named_parameters()}
#     buffers = {k: v.detach() for k, v in model.named_buffers()}

#     def compute_label_score(params, buffers, sample, target):
#         batch = sample.unsqueeze(0)
#         targets = target.unsqueeze(0)
#         predictions = functional_call(model, (params, buffers), (batch, targets))
#         return predictions
    
#     ft_compute_all_label_scores_single_x = vmap(compute_label_score, in_dims=(None, None, None, 0))
#     ft_compute_all_label_scores = vmap(ft_compute_all_label_scores_single_x, in_dims=(None, None, 0, 0))
    
#     ft_compute_label_scores = vmap(compute_label_score, in_dims=(None, None, 0, 0))
#     ft_compute_grad = grad(compute_label_score)
#     ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

#     for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
#         batch_x = batch_x.to(device).reshape(batch_x.shape[0], -1)
#         batch_y = batch_y.to(device)

#         label_selector = torch.as_tensor([ list(range(10)) for i in range(BATCH_SIZE) ])

#         logits = ft_compute_all_label_scores(params, buffers, batch_x, label_selector)
#         loss = criterion(logits, batch_y)
#         print(loss.item())

#         per_sample_grads = ft_compute_sample_grad(params, buffers, batch_x, batch_y)
#         per_sample_scores = ft_compute_label_scores(params, buffers, batch_x, batch_y)

#         for k in per_sample_grads:
#             grads = per_sample_grads[k]
#             dims = grads.shape

#             # norms = torch.linalg.norm(grads, dim=[d for d in range(1, len(dims))])
#             # scale = torch.max(norms**2 + per_sample_scores, torch.as_tensor(1.0))
#             # update = torch.mean(grads * scale.reshape(*([-1] + [1 for d in range(1, len(dims))])), dim=0)

#             # params[k] += lr * update
#             params[k] -= lr * torch.mean(grads, dim=0)