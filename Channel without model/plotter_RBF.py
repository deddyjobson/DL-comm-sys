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
parser.add_argument('--n',type=int,default=4) # encoding length
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--SNR_dB',type=float,default=15) # signal to noise ratio
parser.add_argument('--e_prec',type=int,default=5) # precision of error

hyper = parser.parse_args()
hyper.M = 2 ** hyper.k
hyper.SNR = 10 ** (hyper.SNR_dB/10)


device = "cpu" # default

def generate_input(amt=hyper.bs): # to generate inputs
    indices = torch.randint(low=0,high=hyper.M,size=(amt,),device=device)
    return indices,torch.eye(hyper.M,device=device)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100

def channel_output(enc): # enc.size() = [-1,2*hp.n]
    coeffs = torch.sqrt(torch.rand_like(enc,device=device)**2+torch.rand_like(enc,device=device)**2)/2**0.5
    noise = torch.randn_like(enc,device=device) / scaler
    return coeffs*enc + noise

encoder = torch.load( join('Best','best_encoder_RBF({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) ).to(device)
dec_params = torch.load( join('Best','best_decoder_RBF({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) ).to(device)
def decoder(x):
    y = decoder.lin1(x)
    y = torch.tanh(y)
    y = decoder.lin2(y)
    y = torch.cat((x,y),dim=1)
    y = decoder.lin3(y)
    y = torch.nn.ReLU()(y)
    y = decoder.lin4(y)
    return y
decoder.lin1 = dec_params[0]
decoder.lin2 = dec_params[1]
decoder.lin3 = dec_params[2]
decoder.lin4 = dec_params[3]


encoder.eval()
dec_params.eval()

low = 0
up = 30
num_pts = (up-low)+1

snr_dBs = np.linspace( low,up,num_pts )
snrs = 10 ** (snr_dBs/10)

errs = np.zeros_like(snrs)

for i,snr in enumerate(snrs):
    print(i)
    scaler = np.sqrt( snr )

    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = channel_output(enc)
    op = decoder(enc)

    errs[i] = error_rate(op,labels)


xx = snr_dBs
yy = errs + 1 / 10**hyper.e_prec # to protect against strange behaviour

plt.figure(dpi=250)
axes = plt.gca()
axes.set_xlim([low,up])
axes.set_ylim([1e-3,1e0])
plt.semilogy(xx,yy,'-b')
plt.semilogy(xx,yy,'or')
plt.xticks([0,10,20,30])
plt.yticks([1,1e-1,1e-2])
plt.title('Error profile for RBF channel ({0},{1})'.format(hyper.n,hyper.k))
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Block Error Rate')
plt.grid()
plt.savefig('error_rates_RBF_({0},{1}).png'.format(hyper.n,hyper.k))
plt.show()

exit()
