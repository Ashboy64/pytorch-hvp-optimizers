# Adapted from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=i4ENBTdulBEI

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


from datasets import load_dataset
from transformers import BertTokenizer, BertModel

from sketchy_sgd import *
from block_sketchy_sgd import * 
from generalized_bsgd import * 
from generalized_bsgd_vectorized import * 
from generalized_bsgd_vectorized_trunc import * 
from sketchy_system_sgd import * 
from agd import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OPTIMIZERS = {'sgd': optim.SGD,
              'adam': optim.Adam, 
              'adamw': optim.AdamW, 
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

METRIC_NAME = "f1"


# Load dataset and tokenizer
dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Tokenizes and forms label matrix for a batch of tweets
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

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Process dataset
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")

    train_dataloader = DataLoader(encoded_dataset["train"], cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(encoded_dataset["validation"], cfg.batch_size, shuffle=False)
    test_dataloader = DataLoader(encoded_dataset["test"], cfg.batch_size, shuffle=False)

    # Load model
    encoder = BertModel.from_pretrained("bert-base-uncased")
    classifier = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, len(labels))
    )

    # Initialize optimizer
    if CONFIG['optimizer'] not in CUSTOM_OPTS:
        optimizer = OPTIMIZERS[CONFIG['optimizer']](classifier.parameters(), CONFIG['lr'])
    else:
        optimizer = OPTIMIZERS[CONFIG['optimizer']](classifier, )

    # Main train loop
    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        
        bert_encoding = encoder(batch['input_ids'])
        print(bert_encoding.pooler_output.shape)

        if batch_idx >= 5:
            break
        # outputs = model(input_ids=batch['input_ids'].to(device), 
        #                 labels=batch['labels'].to(device))
        # train_loss = outputs['loss']

        # optimizer.zero_grad()
        # train_loss.backward()
        # optimizer.step()


if __name__ == '__main__':
    main()


# # Evaluation metrics
# # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
# def multi_label_metrics(predictions, labels, threshold=0.5):
#     # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(torch.Tensor(predictions))
#     # next, use threshold to turn them into integer predictions
#     y_pred = np.zeros(probs.shape)
#     y_pred[np.where(probs >= threshold)] = 1
#     # finally, compute metrics
#     y_true = labels
#     f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
#     roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
#     accuracy = accuracy_score(y_true, y_pred)
#     # return as dictionary
#     metrics = {'f1': f1_micro_average,
#                'roc_auc': roc_auc,
#                'accuracy': accuracy}
#     return metrics

# def compute_metrics(p: EvalPrediction):
#     preds = p.predictions[0] if isinstance(p.predictions, 
#             tuple) else p.predictions
#     result = multi_label_metrics(
#         predictions=preds, 
#         labels=p.label_ids)
#     return result



# # Using HuggingFace trainer
# # Main training call
# args = TrainingArguments(
#     f"bert-finetuned-sem_eval-english",
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
#     #push_to_hub=True,
# )

# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()

# # Evaluate model 
# trainer.evaluate()