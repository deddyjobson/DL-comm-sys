"""
Pruning a MLP by weights with one shot
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
import os
import errno
import matplotlib.pyplot as plt

from time import time
from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from os.path import join
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=20) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=64) # batch size
parser.add_argument('--n',type=int,default=8) # number of channels
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--depth',type=int,default=1) # number of hidden layers
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-3) # learning rate
parser.add_argument('--pp',type=float,default=30) # pruning percentage (/100)
parser.add_argument('--SNR',type=float,default=4) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--gpu',type=int,default=1)


hp = parser.parse_args()
hp.M = 2 ** hp.k # number of messages
scaler = np.sqrt( hp.SNR * 2 * hp.k / hp.n )
SNR_dB = 10 * np.log10(hp.SNR)
start = time()


device = "cpu" # default
if hp.gpu and torch.cuda.is_available():
    device = "cuda:0"

def weights_init(m):
    if isinstance(m, MaskedLinear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=hp.init_std) # STD TERM PRETTY SENSITIVE

def log(t,loss,acc):
    log.mov_acc = log.mov_mom * log.mov_acc + acc
    den = 1 - log.mov_mom
    temp = log.mov_acc * (1 - log.mov_mom) / (1 - log.mov_mom**(t+1))
    print('{0}\tLoss:{1:.4e}\tAcc:{2:.2f}%\tMoving:{3:.2f}%'.format(t,loss.item(),acc,temp))
log.mov_acc = 0
log.mov_mom = 0.95

def generate_input(amt=hp.bs): # to generate inputs
    indices = torch.randint(low=0,high=hp.M,size=(amt,),device=device)
    return indices,torch.eye(hp.M,device=device)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100


def train():
    encoder.train()
    decoder.train()
    for t in range(hp.n_epochs):
        for _ in range(hp.n_batches):
            labels,ip = generate_input()
            enc = encoder(ip)
            enc = enc + torch.randn_like(enc,device=device) / scaler
            op = decoder(enc)

            loss = loss_fn(op,labels)
            loss.backward() # compute gradients

            optimizer.step() # update parameters
            optimizer.zero_grad()
        if hp.verbose >= 1:
            acc = accuracy(op,labels)
            log(t,loss,acc)

def test():
    encoder.eval()
    decoder.eval()
    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss with encoding:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )
    return acc

scal = np.sqrt( 5 * 2 * hp.k / hp.n ) # to better see the effect of extreme pruning
errs = []
xx = []

# for prun_rate in range(0,100,11):
for prun_rate in list(np.linspace(0,85,21))+list(np.linspace(86,99,17)):
    encoder = torch.load(join('Best','best_encoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    decoder = torch.load(join('Best','best_decoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    encoder.to(device)
    decoder.to(device)
    net = torch.nn.Sequential(encoder,decoder)
    net.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=hp.lr)

    print("--- Pretrained network loaded ---")
    # test()

    # prune the weights
    masks = weight_prune(net, prun_rate)
    i = 0
    for part in net: # part in [encoder,decoder]
        for p in part[::2]: # conveniently skips biases
            p.set_mask(masks[i])
            i+=1
    print("--- {}% parameters pruned ---".format(prun_rate))
    # test()

    # Retraining
    train()

    # Check accuracy and nonzeros weights in each layer
    print("--- After retraining ---")
    # acc = test()
    actual_perc = prune_rate(net)

    #evaluation
    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scal
    op = decoder(enc)

    errs.append( error_rate(op,labels) )
    xx.append(actual_perc)

errs.append(1.0)
xx.append(100) # duh!
errs = np.array(errs)

# now work with regularized
errs_r = []
xx_r = []
for prun_rate in list(np.linspace(0,85,21))+list(np.linspace(86,99,17)):
    # encoder = torch.load(join('Best','best_encoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    encoder = torch.load(join('Best','regu_encoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    # decoder = torch.load(join('Best','best_decoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    decoder = torch.load(join('Best','regu_decoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    encoder.to(device)
    decoder.to(device)
    net = torch.nn.Sequential(encoder,decoder)
    net.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=hp.lr)

    print("--- Pretrained network loaded ---")
    # test()

    # prune the weights
    masks = weight_prune(net, prun_rate)
    i = 0
    for part in net: # part in [encoder,decoder]
        for p in part[::2]: # conveniently skips biases
            p.set_mask(masks[i])
            i+=1
    print("--- {}% parameters pruned ---".format(prun_rate))
    # test()

    # Retraining
    train()

    # Check accuracy and nonzeros weights in each layer
    print("--- After retraining ---")
    # acc = test()
    actual_perc = prune_rate(net)

    #evaluation
    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scal
    op = decoder(enc)

    errs_r.append( error_rate(op,labels) )
    xx_r.append(actual_perc)

errs_r.append(1.0)
xx_r.append(100) # duh!
errs_r = np.array(errs_r)



try:
    os.makedirs('Pruning Sweep')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

plt.figure(dpi=250)
axes = plt.gca()
axes.set_xlim([0,100])
axes.set_ylim([1e-3,1e0])
plt.semilogy(xx,errs + 1 / 10**hp.e_prec,'b',label='vanilla')
plt.semilogy(xx_r,errs_r + 1 / 10**hp.e_prec,'g',label='regularized')

plt.title('Pruned autoencoder ({0},{1})'.format(hp.n,hp.k))
plt.xlabel('Percentile of weights removed')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend()
plt.savefig(join( 'Pruning Sweep','error_cm_({0},{1}).png'.format(hp.n,hp.k) ))
plt.show()
