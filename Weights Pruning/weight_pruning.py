"""
Pruning a MLP by weights with one shot
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from models import MLP


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=1000) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=64) # batch size
parser.add_argument('--n',type=int,default=7) # number of channels
parser.add_argument('--k',type=int,default=4) # number of bits
parser.add_argument('--depth',type=int,default=1) # number of hidden layers
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-3) # learning rate
parser.add_argument('--SNR',type=float,default=2.51188643150958) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--load_best',type=int,default=0) # whether to load best model
parser.add_argument('--train',type=int,default=1) # whether to train
parser.add_argument('--inspect',type=int,default=1) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=1) # to visualize encodings
parser.add_argument('--boundaries',type=int,default=1) # to visualize encodings

hyper = parser.parse_args()
# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 5,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}


# Data loaders
train_dataset = datasets.MNIST(root='../data/',train=True, download=True,
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset,
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=param['test_batch_size'], shuffle=True)


# Load the pretrained model
net = MLP()
net.load_state_dict(torch.load('models/mlp_pretrained.pkl'))
if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
print("--- Pretrained network loaded ---")
test(net, loader_test)

# prune the weights
masks = weight_prune(net, param['pruning_perc'])
net.set_masks(masks)
print("--- {}% parameters pruned ---".format(param['pruning_perc']))
test(net, loader_test)


# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
                                weight_decay=param['weight_decay'])

train(net, criterion, optimizer, param, loader_train)


# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
test(net, loader_test)
prune_rate(net)


# Save and load the entire model
torch.save(net.state_dict(), 'models/mlp_pruned.pkl')
