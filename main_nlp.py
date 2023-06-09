from tqdm import tqdm
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

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from utils import * 
from data import *
from models import * 

from sketchy_sgd import *
from block_sketchy_sgd import * 
from generalized_bsgd import * 
from generalized_bsgd_vectorized import * 
from generalized_bsgd_vectorized_trunc import * 
from sketchy_system_sgd import * 
from agd import * 

from filters import * 

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
print(f"Device: {device}")


DATASETS = {'mnist': load_mnist, 
            'cifar-10': load_cifar10, 
            'rcv1': load_rcv1, 
            'fashion-mnist': load_fashion_mnist, 
            'sentiment': load_sentiment,
            'bert-encoded-sentiment': load_bert_encoded_sentiment,
            'yelp': load_yelp}
MODELS = {'mlp': MLP, 'cnn': ConvNet, 'bert_encoded_mlp': BertEncodedMLP}

OPTIMIZERS = {'sgd': optim.SGD,
              'adam': optim.Adam, 
              'adamw': optim.AdamW, 
              'agd': AGD, 
              'sketchy_sgd': SketchySGD,
              'block_sketchy_sgd': BlockSketchySGD, 
              'generalized_bsgd': GeneralizedBSGD,
              'generalized_bsgd_vectorized': GeneralizedBSGDVectorized,
              'generalized_bsgd_vectorized_trunc': GeneralizedBSGDVectorizedTrunc,
              'sketchy_system_sgd': SketchySystemSGD
              }

CUSTOM_OPTS = ['agd', 'sketchy_sgd', 
               'block_sketchy_sgd', 'sketchy_system_sgd',
               'generalized_bsgd', 'generalized_bsgd_vectorized', 
               'generalized_bsgd_vectorized_trunc']

FILTERS = {'identity': IdentityFilter, 'momentum': MomentumFilter}

LOSSES = {
    'ce': nn.CrossEntropyLoss(),
    'bce': nn.BCEWithLogitsLoss()
    # 'bce': nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(5.))
}


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions)).cpu()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels.cpu()
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'acc': accuracy}
    return metrics


# @torch.no_grad()
# def evaluate_single(model, dataloader, criterion):
#     loss = 0

#     print("Running evaluation.")

#     y_logits = []
#     y_true = []
#     for batch_idx, batch in enumerate(tqdm(dataloader)):
#     # for batch in dataloader:
#         if type(batch) in [tuple, list] and len(batch) == 2:
#             batch_x = batch[0].to(device)
#             batch_y = batch[1].to(device)
#         else:
#             batch_x = batch['input_ids'].to(device)
#             batch_y = batch['stars'].to(device) - 1
        
#         logits = model(batch_x)
#         loss += criterion(logits, batch_y).item() * batch_x.shape[0]

#         y_logits.append(logits)
#         y_true.append(batch_y)

#         if batch_idx > 3: break

#     y_logits = torch.concat(y_logits, dim=0)
#     y_true = torch.concat(y_true, dim=0)

#     # metrics = multi_label_metrics(y_logits, y_true)
#     metrics = {}
#     loss /= (len(dataloader) * batch_x.shape[0])
#     metrics['loss'] = loss

    # return metrics


@torch.no_grad()
def evaluate_single(model, dataloader, criterion):
    val_loss = 0
    num_correct = 0
    num_total = len(dataloader.dataset)

    # print("Running evaluation.")
    # for batch in tqdm(dataloader):
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if type(batch) in [tuple, list] and len(batch) == 2:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
        else:
            batch_x = batch['input_ids'].to(device)
            batch_y = batch['stars'].to(device) # - 1
        
        logits = model(batch_x)
        val_loss += criterion(logits, batch_y).item() * batch_x.shape[0]

        if len(batch_y.shape) == 1:
            num_correct += (logits.argmax(1) == batch_y).sum().item()
        else:
            preds = (logits > 0.).float()
            num_correct = torch.all(preds == batch_y, dim=1).sum().item()

            # num_correct += (preds == batch_y).sum().item()

            # print(preds)
            # print(batch_y)
            
            # num_total corresponds to the total number of elements in a grid 
            # with len(dataloader.dataset) rows and batch_y.shape[1] columns.
            # num_total is initialized to be the number of elements in one col,
            # here we add back the rest to get the correct num_total
            # num_total += batch_y.shape[0] * (batch_y.shape[1] - 1)

    val_loss /= (len(dataloader) * batch_x.shape[0])
    val_accuracy = num_correct / num_total 
    metrics = {
        'loss': val_loss,
        'acc': val_accuracy
    }

    return metrics


@torch.no_grad()
def evaluate(model, val_loader, test_loader, criterion):
    val_metrics = evaluate_single(model, val_loader, criterion)
    test_metrics = evaluate_single(model, test_loader, criterion)
    return val_metrics, test_metrics


def train(model, train_loader, val_loader, test_loader, opt_config, filter_config, criterion, num_epochs=2, verbose=False):
    opt_name = opt_config.name
    filter_name = filter_config.name
    
    param_dims = [p.shape for p in model.parameters()]
    filterer = FILTERS[filter_name](param_dims, **filter_config.params, device=device)

    if opt_name not in CUSTOM_OPTS:
        optimizer = OPTIMIZERS[opt_name](model.parameters(), **opt_config.params)
    else:
        optimizer = OPTIMIZERS[opt_name](model, **opt_config.params, device=device, filterer=filterer)

    running_loss = 0.0
    avg_val_acc_over_epoch = 0.
    avg_test_acc_over_epoch = 0.

    timestep = 0
    
    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader)):
        # for batch_idx, batch in enumerate(train_loader):
            if type(batch) in [tuple, list] and len(batch) == 2:
                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)
            else:
                batch_x = batch['input_ids'].to(device)
                batch_y = batch['stars'].to(device) # - 1
            
            logits = model(batch_x)
            train_loss = criterion(logits, batch_y)

            if torch.any(torch.isnan(train_loss)):
                print("Loss is nan, terminating run")
                return

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
                val_metrics, test_metrics = evaluate(model, val_loader, test_loader, criterion)
                val_acc = val_metrics['acc']
                test_acc = test_metrics['acc']

                avg_val_acc_over_epoch = (epoch_idx*avg_val_acc_over_epoch + val_acc) / (epoch_idx + 1)
                avg_test_acc_over_epoch = (epoch_idx*avg_test_acc_over_epoch + test_acc) / (epoch_idx + 1)
                
                to_log['timestep'] = timestep
                # to_log['val_acc_over_epoch'] = val_acc 
                
                for val_k in val_metrics:
                    to_log[f'val/{val_k}_over_epoch'] = val_metrics[val_k]
                for test_k in test_metrics:
                    to_log[f'test/{test_k}_over_epoch'] = test_metrics[test_k]
                
                to_log['val/avg_val_acc_over_epoch'] = avg_val_acc_over_epoch
                to_log['test/avg_test_acc_over_epoch'] = avg_test_acc_over_epoch
                
                # to_log['test_acc_over_epoch'] = test_acc 

            if batch_idx % 250 == 0:
                if len(to_log) == 0:
                    val_metrics, test_metrics = evaluate(model, val_loader, test_loader, criterion)
                    val_acc = val_metrics['acc']
                    test_acc = test_metrics['acc']

                    to_log['timestep'] = timestep

                    for val_k in val_metrics:
                        to_log[f'val/{val_k}'] = val_metrics[val_k]
                    for test_k in test_metrics:
                        to_log[f'test/{test_k}'] = test_metrics[test_k]

                to_log['loss'] = train_loss

                for k in step_info:
                    if k in ['avg_lam_hats', 'avg_step_mags', 'avg_step_rms', 'avg_lrs', 'avg_step_deviations', 'avg_grad_sims']:
                        to_log[k] = np.mean(step_info[k])
                    else:
                        to_log[k] = step_info[k]

                if verbose:
                    out_str = f'Epoch {epoch_idx} Batch num {batch_idx}: '
                    out_str = out_str + f'train_loss={running_loss / 100:.3f}, '
                    out_str = out_str + f'val_loss={val_metrics["loss"]:.3f}, val_acc={val_acc:.3f}, '
                    out_str = out_str + f'test_loss={test_metrics["loss"]:.3f}, test_acc={test_acc:.3f}, '
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
    
    train_loader, val_loader, test_loader, dataset_info = DATASETS[cfg.dataset](cfg.batch_size)

    print(f"train set size: {len(train_loader)}")
    print(f"val set size: {len(val_loader)}")
    print(f"test set size: {len(test_loader)}")

    model = MODELS[cfg.model](dataset_info).to(device)
    loss_fn = LOSSES[cfg.loss]
    
    total_start_time = time()
    train(model, 
          train_loader, val_loader, test_loader,
          num_epochs=cfg.num_epochs, 
          opt_config=cfg.optimizer, 
          filter_config=cfg.filter,
          criterion=loss_fn,
          verbose=True)

    # final_val_perf = evaluate(model, val_loader, test_loader)
    print(f"Total time taken for run: {time() - total_start_time}")

    # Flush logs
    wandb.finish()


if __name__ == '__main__':
    main()

