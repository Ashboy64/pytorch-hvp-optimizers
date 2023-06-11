import math
import torch 
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 

from transformers import AutoModel


class MLP(nn.Module):
    # def __init__(self, input_dim, num_classes, hidden_sizes=[128, 64, 32, 16]):
    # def __init__(self, dataset_info, hidden_sizes=[100, 100, 100]): USE FOR FASHION MNIST RESULTS
    def __init__(self, dataset_info, hidden_sizes=[]): # USE FOR BERT_ENCODED_SENTIMENT RESULTS
        super().__init__()
        self.input_dim = dataset_info['input_dim']
        num_inputs = 1 
        for dim in self.input_dim:
            num_inputs *= dim
        
        layers = []
        last_out_size = num_inputs
        for h_size in hidden_sizes:
            layers.append( nn.Linear(last_out_size, h_size) )
            layers.append( nn.ReLU() )
            # layers.append( nn.Sigmoid() )
            last_out_size = h_size
        layers.append( nn.Linear(last_out_size, dataset_info['num_classes']) )

        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.model(x)


class ConvNet(nn.Module):
    def __init__(self, dataset_info):
        super().__init__()

        assert dataset_info['input_dim'] == (3, 32, 32), \
            'ConvNet currently supports only 3x32x32 images'
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dataset_info['num_classes'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BertEncodedMLP:
    def __init__(self, dataset_info, encoder_model="distilbert-base-uncased", freeze_encoder=False, 
                 encoding_dim=768, hidden_sizes=[]):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            print(f"FREEZING ENCODER")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # 100 param groups in distilbert
            num_to_train = 3
            for p_idx, p in enumerate(self.encoder.parameters()):
                if p_idx <= 99 - num_to_train:
                    p.requires_grad = False 
                else:
                    print(f"Setting param {p_idx} to train. Shape: {p.shape}")
                    p.requires_grad = True
                # print(p_idx)

            self.encoder.train()

        classifier_data_info = {'input_dim': (encoding_dim,), 'num_classes': dataset_info['num_classes']}
        self.classifier = MLP(classifier_data_info, hidden_sizes)
    
    def train(self):
        if not self.freeze_encoder:
            self.encoder.train()
        self.classifier.train()
    
    def eval(self):
        self.encoder.eval()
        self.classifier.eval()
    
    def parameters(self):
        if self.freeze_encoder:
            return self.classifier.parameters()
        return list(self.classifier.parameters()) + list(self.encoder.parameters())
    
    def __call__(self, x):
        # x = self.encoder(x).pooler_output
        x = self.encoder(x).last_hidden_state[:, 0, :]
        return self.classifier(x)
    
    def to(self, device):
        self.encoder.to(device)
        self.classifier.to(device)
        return self
