import numpy as np
import argparse
import torch
import pickle
import os
import errno
import matplotlib.pyplot as plt

from shutil import copyfile
from os.path import join
from scipy.stats.mstats import gmean


parser = argparse.ArgumentParser()
parser.add_argument('--bs',type=int,default=32) # batch size
parser.add_argument('--n',type=int,default=2) # encoding length
parser.add_argument('--k',type=int,default=2) # number of bits
parser.add_argument('--SNR',type=float,default=2.51188643150958) # signal to noise ratio
# parser.add_argument('--SNR',type=float,default=1.5848931924611136) # signal to noise ratio
parser.add_argument('--e_prec',type=int,default=6) # precision of error

hyper = parser.parse_args()
hyper.M = 2 ** hyper.k


device = "cpu" # default

def generate_input(amt=hyper.bs): # to generate inputs
    indices = torch.randint(low=0,high=hyper.M,size=(amt,),device=device)
    return indices,torch.eye(hyper.M,device=device)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100


encoder = torch.load( join('Best','best_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
decoder = torch.load( join('Best','best_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
encoder.to(device)
decoder.to(device)
model = torch.nn.Sequential(encoder,decoder) # end to end model
model.to(device)


# snr_dBs = np.linspace(-2, 10, 25)
# snr_dBs = np.linspace(-4, 8, 25)
low = -2
up = 10

snr_dBs = np.linspace(low,up,25)
snrs = 10 ** (snr_dBs/10)

errs = np.zeros_like(snrs)

for i,snr in enumerate(snrs):
    print(i)
    # scaler = np.sqrt( snr )
    scaler = np.sqrt( snr * 2 * hyper.k / hyper.n )


    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = decoder(enc)

    errs[i] = error_rate(op,labels)


xx = snr_dBs
yy = errs + 1 / 10**hyper.e_prec # to protect against anomalous behaviour

plt.figure(dpi=250)
axes = plt.gca()
axes.set_xlim([low,up])
axes.set_ylim([1e-5,1e0])
plt.semilogy(xx,yy,'-b')
plt.semilogy(xx,yy,'or')
plt.title('Error profile of autoencoder ({0},{1})'.format(hyper.n,hyper.k))
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Block Error Rate')
plt.grid()
plt.savefig('error_rates_({0},{1}).png'.format(hyper.n,hyper.k))
plt.show()

exit()
