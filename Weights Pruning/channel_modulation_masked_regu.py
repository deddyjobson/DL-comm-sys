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
from sklearn.decomposition import PCA
from time import time
from pruning.layers import MaskedLinear


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=1000) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=256) # batch size
parser.add_argument('--n',type=int,default=8) # number of channels
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--depth',type=int,default=1) # number of hidden layers
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=3e-4) # learning rate
parser.add_argument('--SNR',type=float,default=4) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--load_best',type=int,default=0) # whether to load best model
parser.add_argument('--train',type=int,default=1) # whether to train
parser.add_argument('--inspect',type=int,default=0) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=0) # to visualize encodings
parser.add_argument('--regu',type=float,default=3e-4) # regularization term

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


if False:
    encoder = torch.load( join('Best','regu_encoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) )
    decoder = torch.load( join('Best','regu_decoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) )
    encoder.to(device)
    decoder.to(device)
    model = torch.nn.Sequential(encoder,decoder) # end to end model
    model.to(device)

    xx = encoder[0].weight.data.cpu().numpy().reshape(-1)
    plt.figure(dpi=300)
    plt.hist(xx,density=True)
    plt.title('Distribution of weights in layer 1 (regularized)')
    plt.savefig('temp.png')
    plt.show()
    print(xx.shape)
    exit()


if hp.load_best:
    encoder = torch.load( join('Best','regu_encoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) )
    decoder = torch.load( join('Best','regu_decoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) )
    encoder.to(device)
    decoder.to(device)
    model = torch.nn.Sequential(encoder,decoder) # end to end model
    model.to(device)
else: # create the network
    hiddens = []
    for _ in range(hp.depth):
        hiddens.append(MaskedLinear(hp.M, hp.M))
        hiddens.append(torch.nn.ReLU())

    encoder = torch.nn.Sequential(
        *hiddens,
        MaskedLinear(hp.M, hp.n),
        torch.nn.BatchNorm1d(hp.n, affine=False) # contrains power of transmitter
    )

    hiddens = []
    for _ in range(hp.depth-1):
        hiddens.append(MaskedLinear(hp.M, hp.M))
        hiddens.append(torch.nn.ReLU())

    decoder = torch.nn.Sequential(
        MaskedLinear(hp.n, hp.M), torch.nn.ReLU(),
        *hiddens,
        MaskedLinear(hp.M, hp.M)
    )

    encoder.to(device)
    decoder.to(device)
    model = torch.nn.Sequential(encoder,decoder) # end to end model
    model.to(device)

    model.apply(weights_init)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
regularizer = torch.nn.L1Loss(reduction='sum')



if hp.train:
    for t in range(hp.n_epochs):
        for _ in range(hp.n_batches):
            labels,ip = generate_input()
            enc = encoder(ip)
            enc = enc + torch.randn_like(enc,device=device) / scaler
            op = decoder(enc)

            loss = loss_fn(op,labels)
            reg_loss = 0 # making regularization term
            for param in model.parameters():
                reg_loss += regularizer(param,0*param)
            # reg_loss = sum(map(regularizer,model.parameters()))

            loss += hp.regu * reg_loss
            loss.backward() # compute gradients

            optimizer.step() # update parameters
            optimizer.zero_grad()
        if hp.verbose >= 1:
            acc = accuracy(op,labels)
            log(t,loss,acc)
        if hp.verbose >= 2:
            print(next(model.parameters())[0][0:3])


encoder.eval()
model.eval()
if hp.verbose >= 0:
    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss with encoding:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )


# saving if best performer
if not hp.load_best:
    try:
        os.makedirs('Best')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    candidate = acc.cpu().detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best','regu_acc({1},{2})_{0:.2f}.txt').format(hp.SNR,hp.n,hp.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best','regu_acc({1},{2})_{0:.2f}.txt').format(hp.SNR,hp.n,hp.k) , np.array([candidate]))
        copyfile('channel_modulation_masked.py', join('Best','regu_channel_modulation_masked({1},{2})_{0}.py'.format(hp.SNR,hp.n,hp.k)) )
        torch.save(encoder, join('Best','regu_encoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
        torch.save(decoder, join('Best','regu_decoder({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


if hp.inspect: # to view encodings, etc.
    for _ in range(1):
        labels,ip = generate_input(amt=1)
        print('Input:\t\t',ip.data.cpu().numpy()[0])
        enc = encoder(ip)
        print('Encoding:\t',enc.data.cpu().numpy()[0])
        enc = enc + torch.randn_like(enc,device=device) / scaler
        print('Channel:\t',enc.data.cpu().numpy()[0])
        op = decoder(enc)
        print('Output:\t\t',torch.softmax(op,dim=1).data.cpu().numpy()[0])


if hp.constellation: # to visualize encodings, etc.
    try:
        os.makedirs('Constellations')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ip = torch.eye(hp.M,device=device)
    enc = encoder(ip).cpu().detach().numpy()

    enc_emb = TSNE().fit_transform(enc).T
    enc_emb -= enc_emb.mean(axis=1).reshape(2,1)
    enc_emb /= enc_emb.std()

    plt.figure(dpi=250)
    plt.grid()
    plt.scatter(enc_emb[0],enc_emb[1])
    plt.title('Constellation of autoencoder ({0},{1})'.format(hp.n,hp.k))
    plt.savefig( join('Constellations','constellation_({0},{1}).png'.format(hp.n,hp.k)) )
    plt.show()




print( 'Total time taken:{0:.2f} seconds'.format(time()-start) )
