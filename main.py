import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import random
import numpy as np
from time import time

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from utils import * 
from data import *
from models import * 
from block_sketchy_sgd import * 
from filters import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OPTIMIZERS = {'adam': optim.Adam, 'block_sketchy_sgd': BlockSketchySGD}
FILTERS = {'identity': IdentityFilter, 'momentum': MomentumFilter}
DATASETS = {'mnist': load_mnist}


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

    val_loss /= len(val_loader) * batch_x.shape[0]
    val_accuracy = num_correct / num_total 

    return val_loss, val_accuracy


def train(model, train_loader, val_loader, opt_config, filter_config, num_epochs=2, verbose=False):
    criterion = nn.CrossEntropyLoss()
    opt_name = opt_config.name
    filter_name = filter_config.name
    
    param_dims = [p.shape for p in model.parameters()]
    filterer = FILTERS[filter_name](param_dims, **filter_config.params)

    if opt_name != 'block_sketchy_sgd':
        optimizer = OPTIMIZERS[opt_name](model.parameters(), **opt_config.params)
    else:
        optimizer = OPTIMIZERS[opt_name](model, **opt_config.params, filterer=filterer)

    running_loss = 0.0      # Avg loss over the past 100 samples

    out_timesteps = []
    out_train_loss = []
    out_val_loss = []
    out_val_acc = []

    timestep = 0
    start_timestamp = time()
    
    # We periodically evaluate on validation data. We don't want the time taken to 
    # do so to factor into the logged wall clock time. So we accumulate the time 
    # taken in the val evaluations in time_to_subtract and subtract this out before 
    # logging.
    time_to_subtract = 0.0
    
    for epoch_idx in range(num_epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            to_log = {}
            to_log['timestep'] = timestep
            
            batch_x = batch_x.to(device).reshape(batch_x.shape[0], -1)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            to_log['loss'] = loss

            optimizer.zero_grad()
            if opt_name != 'block_sketchy_sgd':
                loss.backward()
                optimizer.step()
            else:
                optimizer.step(loss)

            running_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                val_start_time = time()

                val_loss, val_acc = evaluate(model, val_loader)
                to_log['val_loss'] = val_loss
                to_log['val_acc'] = val_acc

                if verbose:
                    out_str = f'Epoch {epoch_idx} Batch num {batch_idx}: '
                    out_str = out_str + f'train_loss={running_loss / 100:.3f}, val_loss={val_loss:.3f}, '
                    out_str = out_str + f'val_acc={val_acc:.3f}'
                    print(out_str)

                out_timesteps.append(timestep)
                out_train_loss.append(running_loss / 100)
                out_val_loss.append(val_loss)
                out_val_acc.append(val_acc)

                running_loss = 0.0

                time_to_subtract += time() - val_start_time

            to_log['wall_clock_time'] = time() - start_timestamp - time_to_subtract
            wandb.log(to_log)
            timestep += 1

    return out_timesteps, out_train_loss, out_val_loss, out_val_acc


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Setup W&B logging
    experiment_name = f"{cfg.optimizer.name}_{cfg.main.dataset_name}_{cfg.filter.name}_filter_seed_{cfg.main.seed}_lr_{cfg.optimizer.params.lr}"
    log_config_dict = {
        "opt_name": cfg.optimizer.name,
        "filter_name": cfg.filter.name,
        "dataset": cfg.main.dataset_name, 
        "batch_size": cfg.main.batch_size,
        "num_epochs": cfg.main.num_epochs,
        "seed": cfg.main.seed, 
        "lr": cfg.optimizer.params.lr,
        "full_config": cfg
    }

    wandb.init(
        project = "ee364b-final-project",
        name    = experiment_name,
        entity  = "ee364b-final-project",
        config  = log_config_dict
    )

    # Run experiment
    seed(cfg.main.seed)
    
    train_loader, val_loader, test_loader = DATASETS[cfg.main.dataset_name](cfg.main.batch_size)
    model = MLP(28*28, 10).to(device)
    
    total_start_time = time()
    train(model, train_loader, val_loader, 
          num_epochs=cfg.main.num_epochs, 
          opt_config=cfg.optimizer, 
          filter_config=cfg.filter,
          verbose=True)

    final_val_perf = evaluate(model, val_loader)
    print(f"Total time taken for run: {time() - total_start_time}")

    # Flush logs
    wandb.finish()


if __name__ == '__main__':
    main()

