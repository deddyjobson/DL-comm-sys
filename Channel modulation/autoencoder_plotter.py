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
parser.add_argument('--n_epochs',type=int,default=1000) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=32) # batch size
parser.add_argument('--M',type=int,default=2) # signal length
parser.add_argument('--n',type=int,default=2) # encoding length
parser.add_argument('--verbose',type=int,default=0) # verbosity
parser.add_argument('--lr',type=float,default=5e-4) # learning rate
parser.add_argument('--SNR',type=float,default=1) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.01) # bias initialization
parser.add_argument('--e_prec',type=int,default=7) # precision of error

hyper = parser.parse_args()

print(hyper.M,hyper.n)

def get_net_error(snr):
    hyper.SNR = snr

    encoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.M, hyper.M), torch.nn.ReLU(),
        torch.nn.Linear(hyper.M, hyper.n),
        torch.nn.BatchNorm1d(hyper.n, affine=False) # contrains power of transmitter
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.n, hyper.M), torch.nn.ReLU(),
        torch.nn.Linear(hyper.M, hyper.M)
    )
    model = torch.nn.Sequential(encoder,decoder) # end to end model to be trained

    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight,gain=1)
            torch.nn.init.normal_(m.bias,std=hyper.init_std) # STD TERM PRETTY SENSITIVE
    model.apply(weights_init)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper.lr)

    def generate_input(amt=hyper.bs):
        indices = torch.randint(low=0,high=hyper.M,size=(amt,))
        return indices,torch.eye(hyper.M)[indices]

    def accuracy(out, labels):
      outputs = torch.argmax(out, dim=1)
      return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()
    def error_rate(out,labels):
        return 1 - accuracy(out,labels)/100
    def err(acc):
        return 1 - acc/100

    for t in range(hyper.n_epochs):
        for _ in range(hyper.n_batches):
            labels,ip = generate_input()
            enc = encoder(ip)
            enc = enc + torch.randn_like(enc) / np.sqrt(hyper.SNR)
            op = decoder(enc)

            loss = loss_fn(op,labels)
            loss.backward() # compute gradients

            optimizer.step() # update parameters
            optimizer.zero_grad()

    # Calculate error rate
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc) / np.sqrt(hyper.SNR)
    op = decoder(enc)

    return error_rate(op,labels)

repeats = 2
snr_dBs = np.linspace(-2,10,20)
snrs = 10 ** (snr_dBs/10)

accs = np.zeros((repeats,len(snrs)))


if False:
    for i in range(repeats):
        for j,snr in enumerate(snrs):
            accs[i,j] = get_net_error(snr)
            print(i,j,accs[i,j])
    pickle.dump(accs,open('error_rates_({0},{1}).pkl'.format(hyper.M,hyper.n),'wb'))
else:
    accs = pickle.load(open('error_rates_({0},{1}).pkl'.format(hyper.M,hyper.n),'rb'))
    if True:
        indices = [-2,-1]
        for i in indices:
            print(i)
            print( list(enumerate(accs[0]))[i] )
            accs[0,i] = get_net_error(snrs[i])
            print( list(enumerate(accs[0]))[i] )
        pickle.dump(accs,open('error_rates_({0},{1}).pkl'.format(hyper.M,hyper.n),'wb'))
        # print( list(enumerate(accs[0])) )
        exit()


xx = snr_dBs
# yy = gmean(accs + 1e-10,axis=0) # to protect against anomalous behaviour
# yy = np.median(accs,axis=0) + 1e-10 # to protect against anomalous behaviour
yy = np.min(accs,axis=0) + 1e-10 # to protect against anomalous behaviour

plt.figure(dpi=250)
plt.semilogy(xx,yy)
plt.title('Error profile of autoencoder ({0},{1})'.format(hyper.M,hyper.n))
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Block Error Rate')
plt.grid()
plt.savefig('error_rates_({0},{1}).png'.format(hyper.M,hyper.n))
plt.show()

exit()
