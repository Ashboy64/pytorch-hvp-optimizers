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
from generalized_bsgd import * 
from sketchy_system_sgd import * 
from agd import * 

from filters import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASETS = {'mnist': load_mnist, 'cifar-10': load_cifar10}
MODELS = {'mlp': MLP, 'cnn': ConvNet}

OPTIMIZERS = {'sgd': optim.SGD,
              'adam': optim.Adam, 
              'adamw': optim.AdamW, 
              'agd': AGD, 
              'block_sketchy_sgd': BlockSketchySGD, 
              'generalized_bsgd': GeneralizedBSGD,
              'sketchy_system_sgd': SketchySystemSGD}

CUSTOM_OPTS = ['agd', 'block_sketchy_sgd', 'generalized_bsgd', 'sketchy_system_sgd']

FILTERS = {'identity': IdentityFilter, 'momentum': MomentumFilter}

@torch.no_grad()
def evaluate(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    num_correct = 0
    num_total = len(val_loader.dataset)

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
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

    if opt_name not in CUSTOM_OPTS:
        optimizer = OPTIMIZERS[opt_name](model.parameters(), **opt_config.params)
    else:
        optimizer = OPTIMIZERS[opt_name](model, **opt_config.params, filterer=filterer)

    running_loss = 0.0
    avg_val_acc_over_epoch = 0.

    timestep = 0
    
    for epoch_idx in range(num_epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            train_loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            if opt_name not in CUSTOM_OPTS:
                train_loss.backward()
                optimizer.step()
                step_info = {} 
            else:
                step_info = optimizer.step(train_loss)

            running_loss += train_loss.item()

            to_log = {}
            if batch_idx == len(train_loader)-1:
                val_loss, val_acc = evaluate(model, val_loader)
                avg_val_acc_over_epoch = (epoch_idx*avg_val_acc_over_epoch + val_acc) / (epoch_idx + 1)
                
                to_log['timestep'] = timestep
                to_log['val_acc_over_epoch'] = val_acc 
                to_log['avg_val_acc_over_epoch'] = avg_val_acc_over_epoch

            if batch_idx % 100 == 0:
                if len(to_log) == 0:
                    val_loss, val_acc = evaluate(model, val_loader)
                    to_log['timestep'] = timestep
                    to_log['val_loss'] = val_loss
                    to_log['val_acc'] = val_acc

                to_log['loss'] = train_loss

                for k in step_info:
                    if k in ['avg_lam_hats', 'avg_step_mags', 'avg_lrs', 'avg_step_deviations', 'avg_grad_sims']:
                        to_log[k] = np.mean(step_info[k])
                    else:
                        to_log[k] = step_info[k]

                if verbose:
                    out_str = f'Epoch {epoch_idx} Batch num {batch_idx}: '
                    out_str = out_str + f'train_loss={running_loss / 100:.3f}, val_loss={val_loss:.3f}, '
                    out_str = out_str + f'val_acc={val_acc:.3f}'
                    print(out_str)
                
                running_loss = 0.0

            if len(to_log) > 0:
                wandb.log(to_log)
            
            timestep += 1


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Setup W&B logging
    experiment_name = f"{cfg.wandb.experiment_name_prefix}_{cfg.optimizer.name}_{cfg.dataset}_{cfg.filter.name}_filter_seed_{cfg.seed}"
    if 'lr' in cfg.optimizer.params:
        experiment_name += f'_lr_{cfg.optimizer.params.lr}'
    
    log_config_dict = {
        "opt_name": cfg.optimizer.name,
        "filter_name": cfg.filter.name,
        "dataset": cfg.dataset, 
        "batch_size": cfg.batch_size,
        "num_epochs": cfg.num_epochs,
        "model": cfg.model,
        "seed": cfg.seed, 
        "full_config": cfg
    }

    if "lr" in cfg.optimizer.params:
        log_config_dict["lr"] = cfg.optimizer.params.lr,

    wandb.init(
        project = cfg.wandb.project,
        name    = experiment_name,
        entity  = "ee364b-final-project",
        config  = log_config_dict, 
        mode    = cfg.wandb.mode
    )

    # Run experiment
    seed(cfg.seed)
    
    train_loader, val_loader, test_loader, info = DATASETS[cfg.dataset](cfg.batch_size)

    print(f"train set size: {len(train_loader)}")
    print(f"val set size: {len(val_loader)}")
    print(f"test set size: {len(test_loader)}")

    model = MODELS[cfg.model](info['input_dim'], 10).to(device)
    
    total_start_time = time()
    train(model, train_loader, val_loader, 
          num_epochs=cfg.num_epochs, 
          opt_config=cfg.optimizer, 
          filter_config=cfg.filter,
          verbose=True)

    final_val_perf = evaluate(model, val_loader)
    print(f"Total time taken for run: {time() - total_start_time}")

    # Flush logs
    wandb.finish()


if __name__ == '__main__':
    main()

