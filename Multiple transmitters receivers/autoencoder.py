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
parser.add_argument('--bs',type=int,default=256) # batch size
parser.add_argument('--n',type=int,default=2) # number of channels
parser.add_argument('--k',type=int,default=2) # number of bits
parser.add_argument('--depth',type=int,default=1) # number of hidden layers
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-4) # learning rate
parser.add_argument('--SNR_dB',type=float,default=4) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=0) # to use gpu if possible
temp = 1
parser.add_argument('--load_best',type=int,default=temp) # whether to load best model
parser.add_argument('--train',type=int,default=1-temp) # whether to train
parser.add_argument('--constellation',type=int,default=temp) # to visualize encodings


hyper = parser.parse_args()
hyper.M = 2 ** hyper.k # number of messages
hyper.SNR = 10 ** (hyper.SNR_dB/10)
scaler = np.sqrt( hyper.SNR * hyper.k / hyper.n )
# scaler = np.sqrt( hyper.SNR * 2 * hyper.k / hyper.n )
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

# make network
if hyper.load_best:
    encoder1 = torch.load( join('Best','best_encoder1({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    decoder1 = torch.load( join('Best','best_decoder1({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    encoder1.to(device)
    decoder1.to(device) # separate networks
    encoder2 = torch.load( join('Best','best_encoder2({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    decoder2 = torch.load( join('Best','best_decoder2({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    encoder2.to(device)
    decoder2.to(device) # separate networks
else:
    hiddens = []
    for _ in range(hyper.depth):
        hiddens.append(torch.nn.Linear(hyper.M, hyper.M))
        hiddens.append(torch.nn.ReLU())

    encoder1 = torch.nn.Sequential(
        *hiddens,
        torch.nn.Linear(hyper.M, 2*hyper.n), # 1 complex number == 2 reals
        torch.nn.BatchNorm1d(2*hyper.n, affine=False) # contrains power of transmitter
    )

    hiddens = []
    for _ in range(hyper.depth-1):
        hiddens.append(torch.nn.Linear(hyper.M, hyper.M))
        hiddens.append(torch.nn.ReLU())

    decoder1 = torch.nn.Sequential(
        torch.nn.Linear(2*hyper.n, hyper.M), torch.nn.ReLU(),
        *hiddens,
        torch.nn.Linear(hyper.M, hyper.M)
    )
    hiddens = []
    for _ in range(hyper.depth):
        hiddens.append(torch.nn.Linear(hyper.M, hyper.M))
        hiddens.append(torch.nn.ReLU())

    encoder2 = torch.nn.Sequential(
        *hiddens,
        torch.nn.Linear(hyper.M, 2*hyper.n),
        torch.nn.BatchNorm1d(2*hyper.n, affine=False) # contrains power of transmitter
    )

    hiddens = []
    for _ in range(hyper.depth-1):
        hiddens.append(torch.nn.Linear(hyper.M, hyper.M))
        hiddens.append(torch.nn.ReLU())

    decoder2 = torch.nn.Sequential(
        torch.nn.Linear(2*hyper.n, hyper.M), torch.nn.ReLU(),
        *hiddens,
        torch.nn.Linear(hyper.M, hyper.M)
    )

    encoder1.to(device)
    decoder1.to(device)
    encoder2.to(device)
    decoder2.to(device)

    encoder1.apply(weights_init)
    decoder1.apply(weights_init)
    encoder2.apply(weights_init)
    decoder2.apply(weights_init)

loss_fn = torch.nn.CrossEntropyLoss()

params = list(encoder1.parameters())+list(encoder2.parameters())+list(decoder1.parameters())+list(decoder2.parameters())

optim = torch.optim.Adam(
params, lr=hyper.lr, weight_decay=hyper.decay
 )

alpha = 0.5 # for first epoch
if hyper.train:
    for t in range(hyper.n_epochs):
        for _ in range(hyper.n_batches):
            labels1,ip1 = generate_input()
            labels2,ip2 = generate_input()
            enc1 = encoder1(ip1)
            enc2 = encoder2(ip2)
            enc1,enc2 = (
                enc1 + enc2 + torch.randn_like(enc1,device=device) / scaler,
                enc1 + enc2 + torch.randn_like(enc2,device=device) / scaler
                ) # must be done simultaneously
            op1 = decoder1(enc1)
            op2 = decoder2(enc2)

            loss1 = loss_fn(op1,labels1)
            loss2 = loss_fn(op2,labels2)
            net_loss = alpha*loss1 + (1-alpha)*loss2
            alpha = torch.autograd.Variable(loss1/(loss1+loss2),requires_grad=False)
            net_loss.backward() # compute gradients

            optim.step() # update parameters
            optim.zero_grad()
        if hyper.verbose >= 1:
            acc1 = accuracy(op1,labels1)
            acc2 = accuracy(op1,labels1)
            log(t,net_loss,(acc1+acc2)/2)
        if hyper.verbose >= 2:
            print(next(model.parameters())[0][0:3])



encoder1.eval()
decoder1.eval()
encoder2.eval()
decoder2.eval()
if hyper.verbose >= 0:
    labels1,ip1 = generate_input(amt=10**hyper.e_prec)
    labels2,ip2 = generate_input(amt=10**hyper.e_prec)
    enc1 = encoder1(ip1)
    enc2 = encoder2(ip2)
    enc1,enc2 = (
        enc1 + enc2 + torch.randn_like(enc1,device=device) / scaler,
        enc1 + enc2 + torch.randn_like(enc2,device=device) / scaler
        ) # must be done simultaneously
    op1 = decoder1(enc1)
    op2 = decoder2(enc2)

    losses = [loss_fn(op1,labels1),loss_fn(op2,labels2)]
    accs = [accuracy(op1,labels1),accuracy(op2,labels2)]

    for i in range(2):
        print('\nFor system {0},'.format(i))
        print( 'Loss with encoding:{0:.4e}'.format( losses[i] ) )
        print( 'Accuracy:{0:.2f}%'.format( accs[i] ) )
        print( 'Error rate:{0:.2e}\n\n'.format( 1-accs[i]/100 ) )


# saving if best performer
if not hyper.load_best:
    try:
        os.makedirs('Best')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    candidate = ( (accs[0]+accs[1])/2 ).cpu().detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best','best_acc({1},{2})_{0:.2f}.txt').format(hyper.SNR,hyper.n,hyper.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best','best_acc({1},{2})_{0:.2f}.txt').format(hyper.SNR,hyper.n,hyper.k) , np.array([candidate]))
        copyfile('autoencoder.py', join('Best','best_autoencoder({1},{2})_{0}.py'.format(hyper.SNR,hyper.n,hyper.k)) )
        torch.save(encoder1, join('Best','best_encoder1({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)))
        torch.save(decoder1, join('Best','best_decoder1({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)))
        torch.save(encoder2, join('Best','best_encoder2({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)))
        torch.save(decoder2, join('Best','best_decoder2({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


if hyper.constellation: # to visualize encodings, etc.
    try:
        os.makedirs('Constellations')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ip = torch.eye(hyper.M,device=device)
    enc1 = encoder1(ip).cpu().detach().numpy()
    enc2 = encoder2(ip).cpu().detach().numpy()
    enc = np.concatenate( (enc1,enc2),axis=0 )

    enc_emb = TSNE().fit_transform(enc).T
    enc_emb -= enc_emb.mean(axis=1).reshape(2,1)
    enc_emb /= enc_emb.std()

    plt.figure(dpi=250)
    plt.grid()
    plt.scatter(enc_emb[0,:hyper.M],enc_emb[1,:hyper.M])
    plt.scatter(enc_emb[0,hyper.M:],enc_emb[1,hyper.M:])
    plt.title('Constellation for the two-user model ({0},{1})'.format(hyper.n,hyper.k))
    plt.savefig( join('Constellations','constellation_({0},{1}).png'.format(hyper.n,hyper.k)) )
    plt.show()


print( 'Total time taken:{0:.2f} seconds'.format(time()-start) )
