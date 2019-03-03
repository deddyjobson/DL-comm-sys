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
parser.add_argument('--n',type=int,default=8) # encoding length
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--SNR_dB',type=float,default=20) # signal to noise ratio
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--kwrd',type=str,default='real') # precision of error

hp = parser.parse_args()
hp.M = 2 ** hp.k
hp.SNR = 10 ** (hp.SNR_dB/10)


device = "cpu" # default

def generate_input(amt=hp.bs): # to generate inputs
    indices = torch.randint(low=0,high=hp.M,size=(amt,),device=device)
    return indices,torch.eye(hp.M,device=device)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100

def channel_output(enc): # enc.size() = [-1,2*hp.n]
    coeff = torch.sqrt(torch.randn((enc.shape[0],1),device=device)**2+
            torch.randn((enc.shape[0],1),device=device)**2)/np.sqrt(2)
    noise = torch.randn_like(enc,device=device) / scaler
    return coeff*enc + noise

encoder = torch.load( join('Best','best_encoder_RBF_{3}({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k,hp.kwrd)) ).to(device)
dec_params = torch.load( join('Best','best_decoder_RBF_{3}({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k,hp.kwrd)) ).to(device)
def decoder(x):
    y = decoder.lin1(x)
    y = torch.tanh(y)
    y = decoder.lin2(y)
    y = x / y
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

low = -5
up = 30
num_pts = (up-low)+1

snr_dBs = np.linspace( low,up,num_pts )
snrs = 10 ** (snr_dBs/10)

errs = np.zeros_like(snrs)

for i,snr in enumerate(snrs):
    print(i)
    scaler = np.sqrt( snr/2 ) # 2: to take into account the real-complex normalization ratio

    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc = channel_output(enc)
    op = decoder(enc)

    errs[i] = error_rate(op,labels)


xx = snr_dBs
yy = errs + 1 / 10**hp.e_prec # to protect against strange behaviour

plt.figure(dpi=250)
axes = plt.gca()
axes.set_xlim([low,up])
axes.set_ylim([1e-3,1e0])
plt.semilogy(xx,yy,'-b')
plt.semilogy(xx,yy,'or')
plt.xticks([0,10,20,30])
plt.yticks([1,1e-1,1e-2])
plt.title('Error profile for RBF ({2}) channel ({0},{1})'.format(hp.n,hp.k,hp.kwrd))
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Block Error Rate')
plt.grid()
plt.savefig('error_rates_RBF_{2}_({0},{1}).png'.format(hp.n,hp.k,hp.kwrd))
plt.show()

exit()
