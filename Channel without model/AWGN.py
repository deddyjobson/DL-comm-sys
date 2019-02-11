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
parser.add_argument('--n_epochs',type=int,default=500) # number of epochs
parser.add_argument('--n_batches_enc',type=int,default=100) # number of batches per epoch
parser.add_argument('--n_batches_dec',type=int,default=1000) # number of batches per epoch
parser.add_argument('--bt',type=int,default=256) # batch size
parser.add_argument('--br',type=int,default=128) # batch size
parser.add_argument('--n',type=int,default=4) # number of channels
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr_enc',type=float,default=1e-4) # learning rate
parser.add_argument('--lr_dec',type=float,default=1e-3) # learning rate
parser.add_argument('--SNR_dB',type=float,default=8) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--plot',type=int,default=1)
temp = 0
parser.add_argument('--load_best',type=int,default=temp) # whether to load best model
parser.add_argument('--train',type=int,default=1-temp) # whether to train
parser.add_argument('--inspect',type=int,default=1-temp) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=temp) # to visualize encodings

parser.add_argument('--sig_pi2',type=float,default=0.02) # to visualize encodings


hyper = parser.parse_args()
hyper.M = 2 ** hyper.k # number of messages
hyper.SNR = 10 ** (hyper.SNR_dB/10)
# scaler = np.sqrt( hyper.SNR * hyper.k / hyper.n )
scaler = np.sqrt( hyper.SNR )
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

def generate_input(amt=hyper.br): # to generate inputs
    indices = torch.randint(low=0,high=hyper.M,size=(amt,),device=device)
    return indices,torch.eye(hyper.M,device=device)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100

def add_channel_noise(enc):
    return enc + torch.randn_like(enc,device=device) / scaler

# make network
if hyper.load_best:
    encoder = torch.load( join('Best','best_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    decoder = torch.load( join('Best','best_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
    encoder.to(device)
    decoder.to(device) # separate networks
else:
    encoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.M, hyper.M),torch.nn.ELU(),
        torch.nn.Linear(hyper.M, 2*hyper.n), #transmitting complex numbers
        torch.nn.BatchNorm1d(2*hyper.n, affine=False) # contrains power of transmitter
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(2*hyper.n, hyper.M), torch.nn.ReLU(),
        torch.nn.Linear(hyper.M, hyper.M)
    )

    encoder.to(device)
    decoder.to(device)

    encoder.apply(weights_init)
    decoder.apply(weights_init)

loss_fn_enc = torch.nn.CrossEntropyLoss(reduction='none')
loss_fn_dec = torch.nn.CrossEntropyLoss()

optim_enc = torch.optim.Adam(encoder.parameters(), lr=hyper.lr_enc, weight_decay=hyper.decay)
optim_dec = torch.optim.Adam(decoder.parameters(), lr=hyper.lr_dec, weight_decay=hyper.decay)

def transmit_sample(enc): # to transmit the encoded signals
    return np.sqrt(1-hyper.sig_pi2)*enc + np.sqrt(hyper.sig_pi2)*torch.randn_like(enc,device=device)

def normalize(enc):
    return enc / torch.norm(enc,dim=1).reshape(-1,1) * np.sqrt(hyper.n) # normalization

def transmitter_step(): # reinforced learning
    labels,ip = generate_input(amt=hyper.bt)
    ft = encoder(ip)
    xp = transmit_sample(ft)
    rec = add_channel_noise(xp)
    op = decoder(rec)

    loss_enc = loss_fn_enc(op,labels)

    xp_fixed = torch.autograd.Variable(xp.data,requires_grad=False)
    log_prob = -torch.norm(xp_fixed - np.sqrt(1-hyper.sig_pi2)*ft,2,1)**2/(2*hyper.sig_pi2) #constants eliminated anyway


    if False:
        vect = torch.dot( - log_prob,loss_enc**3) / hyper.bt
    else:
        vect = torch.dot( - log_prob,loss_enc) / hyper.bt


    optim_enc.zero_grad()
    vect.backward() # compute gradients
    optim_enc.step() # update parameters

def receiver_step(): # supervised learning
    labels,ip = generate_input()
    enc = encoder(ip)
    rec = add_channel_noise(enc)
    # rec.requires_grad=False #Now we can't train E2E
    op = decoder(rec)

    loss_dec = loss_fn_dec(op,labels)
    optim_dec.zero_grad()
    loss_dec.backward() # compute gradients
    optim_dec.step() # update parameters


errs = []
#for epoch 0 - before any training
labels,ip = generate_input(amt=10*hyper.br)
enc = encoder(ip)
enc = add_channel_noise(enc)
op = decoder(enc)

acc = accuracy(op,labels)
loss = loss_fn_dec(op,labels)
errs.append(error_rate(op,labels))
log(0,loss,acc)

if hyper.train:
    for _ in range(10*hyper.n_batches_dec): # initial training
        receiver_step()
    for t in range(1,hyper.n_epochs):
        for _ in range(hyper.n_batches_enc):
            receiver_step()
            pass
        for _ in range(hyper.n_batches_dec):
            transmitter_step()
            pass
        if hyper.verbose >= 1:
            labels,ip = generate_input(amt=10*hyper.br)
            enc = encoder(ip)
            enc = add_channel_noise(enc)
            op = decoder(enc)

            acc = accuracy(op,labels)
            loss = loss_fn_dec(op,labels)
            errs.append(error_rate(op,labels))
            log(t,loss,acc)
        if hyper.verbose >= 2:
            print(next(encoder.parameters())[0][0:3])
            print(next(decoder.parameters())[0][0:3])



encoder.eval()
decoder.eval()
if hyper.verbose >= 0:
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = add_channel_noise(enc)
    op = decoder(enc)
    loss = loss_fn_dec(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss with encoding:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )


if hyper.plot:
    try:
        os.makedirs('Training')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.figure(dpi=250)
    plt.plot(range(hyper.n_epochs),errs,'-b')
    plt.title('Training profile of autoencoder ({0},{1})'.format(hyper.n,hyper.k))
    plt.xlabel('Training Epoch')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.savefig( join('Training','training_({0},{1})_{2}.png'.format(hyper.n,hyper.k,hyper.SNR_dB)) )
    plt.show()

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
        copyfile('AWGN.py', join('Best','best_AWGN({1},{2})_{0}.py'.format(hyper.SNR,hyper.n,hyper.k)) )
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
        enc = add_channel_noise(enc)
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
    plt.title('Constellation for AWGN ({0},{1})'.format(hyper.n,hyper.k))
    plt.savefig( join('Constellations','constellation_({0},{1}).png'.format(hyper.n,hyper.k)) )
    plt.show()


print( 'Total time taken:{0:.2f} seconds'.format(time()-start) )
