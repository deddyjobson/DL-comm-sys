import numpy as np
import argparse
import torch
import pickle
import os
import errno
import matplotlib.pyplot as plt

from shutil import copyfile
from os.path import join
from sklearn.manifold import TSNE
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=1000) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=64) # batch size
parser.add_argument('--n',type=int,default=2) # encoding length
parser.add_argument('--k',type=int,default=2) # number of bits
parser.add_argument('--depth',type=int,default=1) # number of hidden layers
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-3) # learning rate
parser.add_argument('--SNR',type=float,default=2.51188643150958) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=0)
temp = 0
parser.add_argument('--load_best',type=int,default=temp) # whether to load best model
parser.add_argument('--train',type=int,default=1-temp) # whether to train
parser.add_argument('--inspect',type=int,default=1-temp) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=temp) # to visualize encodings

hyper = parser.parse_args()
hyper.M = 2 ** hyper.k # number of messages
scaler = np.sqrt( hyper.SNR * 2 * hyper.k / hyper.n )
SNR_dB = 10 * np.log10(hyper.SNR)
start = time()


device = "cpu" # default
if hyper.gpu and torch.cuda.is_available():
    device = "cuda:0"


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=hyper.init_std) # STD TERM PRETTY SENSITIVE

def log(t,loss,acc):
    log.mov_acc = log.mov_mom * log.mov_acc + acc
    den = 1 - log.mov_mom
    temp = log.mov_acc * (1 - log.mov_mom) / (1 - log.mov_mom**(t+1))
    print('{0}\tLoss:{1:.4e}\tAcc:{2:.2f}%\tMoving:{3:.2f}%'.format(t,loss.item(),acc,temp))
log.mov_acc = 0
log.mov_mom = 0.95

def generate_input(amt=hyper.bs): # to generate inputs
    indices = torch.randint(low=0,high=hyper.M,size=(amt,),device=device)
    return indices,torch.eye(hyper.M,device=device)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100


if hyper.load_best:
    encoder = torch.load( join('Best','best_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    decoder = torch.load( join('Best','best_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    encoder.to(device)
    decoder.to(device)
    model = torch.nn.Sequential(encoder,decoder) # end to end model
    model.to(device)
else: # create the network
    hiddens = []
    for _ in range(hyper.depth):
        hiddens.append(torch.nn.Linear(hyper.M, hyper.M))
        hiddens.append(torch.nn.ReLU())

    encoder = torch.nn.Sequential(
        *hiddens,
        torch.nn.Linear(hyper.M, hyper.n),
        torch.nn.BatchNorm1d(hyper.n, affine=False) # contrains power of transmitter
    )

    hiddens = []
    for _ in range(hyper.depth-1):
        hiddens.append(torch.nn.Linear(hyper.M, hyper.M))
        hiddens.append(torch.nn.ReLU())

    decoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.n, hyper.M), torch.nn.ReLU(),
        *hiddens,
        torch.nn.Linear(hyper.M, hyper.M)
    )

    encoder.to(device)
    decoder.to(device)
    model = torch.nn.Sequential(encoder,decoder) # end to end model
    model.to(device)

    model.apply(weights_init)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr, weight_decay=hyper.decay)

if hyper.train:
    for t in range(hyper.n_epochs):
        for _ in range(hyper.n_batches):
            labels,ip = generate_input()
            enc = encoder(ip)
            enc = enc + torch.randn_like(enc,device=device) / scaler
            op = decoder(enc)

            loss = loss_fn(op,labels)
            loss.backward() # compute gradients

            optimizer.step() # update parameters
            optimizer.zero_grad()
        if hyper.verbose >= 1:
            acc = accuracy(op,labels)
            log(t,loss,acc)
        if hyper.verbose >= 2:
            print(next(model.parameters())[0][0:3])


encoder.eval()
model.eval()
if hyper.verbose >= 0:
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss with encoding:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )


# saving if best performer
if not hyper.load_best:
    try:
        os.makedirs('Best')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    candidate = acc.cpu().detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best','best_acc({1},{2})_{0:.2f}.txt').format(hyper.SNR,hyper.n,hyper.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best','best_acc({1},{2})_{0:.2f}.txt').format(hyper.SNR,hyper.n,hyper.k) , np.array([candidate]))
        copyfile('autoencoder.py', join('Best','best_autoencoder({1},{2})_{0}.py'.format(hyper.SNR,hyper.n,hyper.k)) )
        torch.save(encoder, join('Best','best_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)))
        torch.save(decoder, join('Best','best_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


if hyper.inspect: # to view encodings, etc.
    for _ in range(1):
        labels,ip = generate_input(amt=1)
        print('Input:\t\t',ip.data.cpu().numpy()[0])
        enc = encoder(ip)
        print('Encoding:\t',enc.data.cpu().numpy()[0])
        enc = enc + torch.randn_like(enc,device=device) / scaler
        print('Channel:\t',enc.data.cpu().numpy()[0])
        op = decoder(enc)
        print('Output:\t\t',torch.softmax(op,dim=1).data.cpu().numpy()[0])


if hyper.constellation: # to visualize encodings, etc.
    try:
        os.makedirs('Constellations')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ip = torch.eye(hyper.M,device=device)
    enc = encoder(ip).cpu().detach().numpy()

    enc_emb = TSNE().fit_transform(enc).T
    enc_emb -= enc_emb.mean(axis=1).reshape(2,1)
    enc_emb /= enc_emb.std()

    plt.figure(dpi=250)
    plt.grid()
    plt.scatter(enc_emb[0],enc_emb[1])
    plt.title('Constellation of autoencoder ({0},{1})'.format(hyper.n,hyper.k))
    plt.savefig( join('Constellations','constellation_({0},{1}).png'.format(hyper.n,hyper.k)) )
    plt.show()


print( 'Total time taken:{0:.2f} seconds'.format(time()-start) )
