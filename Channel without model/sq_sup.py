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


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=500) # number of epochs
parser.add_argument('--n_batches_enc',type=int,default=100) # number of batches per epoch
parser.add_argument('--n_batches_dec',type=int,default=1000) # number of batches per epoch
parser.add_argument('--bt',type=int,default=256) # batch size
parser.add_argument('--br',type=int,default=128) # batch size
parser.add_argument('--n',type=int,default=7) # number of channels
parser.add_argument('--k',type=int,default=4) # number of bits
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr_enc',type=float,default=1e-4) # learning rate
parser.add_argument('--lr_dec',type=float,default=1e-3) # learning rate
parser.add_argument('--SNR_dB',type=float,default=6) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--plot',type=int,default=1)
parser.add_argument('--sup',type=int,default=1) # whether to use supervised or rl
temp = 0
parser.add_argument('--load_best',type=int,default=temp) # whether to load best model
parser.add_argument('--train',type=int,default=1-temp) # whether to train
parser.add_argument('--inspect',type=int,default=1-temp) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=0) # to visualize encodings
parser.add_argument('--boundaries',type=int,default=temp) # to visualize encodings

parser.add_argument('--sig_pi2',type=float,default=0.02) # to visualize encodings


hp = parser.parse_args()
hp.M = 2 ** hp.k # number of messages
hp.SNR = 10 ** (hp.SNR_dB/10)
# scaler = np.sqrt( hp.SNR * hp.k / hp.n )
scaler = np.sqrt( hp.SNR )
start = time()


device = "cpu" # default
if hp.gpu and torch.cuda.is_available():
    device = "cuda:0"


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=hp.init_std) # STD TERM PRETTY SENSITIVE

def log(t,loss,acc):
    log.mov_acc = log.mov_mom * log.mov_acc + acc
    den = 1 - log.mov_mom
    temp = log.mov_acc * (1 - log.mov_mom) / (1 - log.mov_mom**(t+1))
    print('{0}\tLoss:{1:.4e}\tAcc:{2:.2f}%\tMoving:{3:.2f}%'.format(t,loss.item(),acc,temp))
log.mov_acc = 0
log.mov_mom = 0.95

def generate_input(amt=hp.br): # to generate inputs
    indices = torch.randint(low=0,high=hp.M,size=(amt,),device=device)
    return indices,torch.eye(hp.M,device=device)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100

# def add_channel_noise(enc):
#     return enc + torch.randn_like(enc,device=device) / scaler

def add_channel_noise(enc):
    with torch.no_grad(): # we're morphing the gradients to suit our needs
        noise = ( enc*enc + torch.randn_like(enc,device=device)/scaler ) - enc
    return enc + noise

# make network
if hp.load_best:
    encoder = torch.load( join('Best','encoder({1},{2})_{0}_{3}.pt'.format(hp.SNR,hp.n,hp.k,hp.sup)) )
    decoder = torch.load( join('Best','decoder({1},{2})_{0}_{3}.pt'.format(hp.SNR,hp.n,hp.k,hp.sup)) )
    encoder.to(device)
    decoder.to(device) # separate networks
else:
    encoder = torch.nn.Sequential(
        torch.nn.Linear(hp.M, hp.M),torch.nn.ELU(),
        torch.nn.Linear(hp.M, 2*hp.n), #transmitting complex numbers
        torch.nn.BatchNorm1d(2*hp.n, affine=False) # contrains power of transmitter
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(2*hp.n, hp.M), torch.nn.ReLU(),
        torch.nn.Linear(hp.M, hp.M)
    )

    encoder.to(device)
    decoder.to(device)

    encoder.apply(weights_init)
    decoder.apply(weights_init)

loss_fn_enc = torch.nn.CrossEntropyLoss(reduction='none')
loss_fn_dec = torch.nn.CrossEntropyLoss()

optim_enc = torch.optim.Adam(encoder.parameters(), lr=hp.lr_enc, weight_decay=hp.decay)
optim_dec = torch.optim.Adam(decoder.parameters(), lr=hp.lr_dec, weight_decay=hp.decay)

def transmit_sample(enc): # to transmit the encoded signals
    return np.sqrt(1-hp.sig_pi2)*enc + np.sqrt(hp.sig_pi2)*torch.randn_like(enc,device=device)

def normalize(enc):
    return enc / torch.norm(enc,dim=1).reshape(-1,1) * np.sqrt(hp.n) # normalization

def transmitter_step_rl(): # reinforced learning
    labels,ip = generate_input(amt=hp.bt)
    ft = encoder(ip)
    with torch.no_grad(): # now, backprop learning from labels impossible
        xp = transmit_sample(ft)
        rec = add_channel_noise(xp)
        op = decoder(rec)

    loss_enc = loss_fn_enc(op,labels)
    log_prob = -torch.norm(xp - np.sqrt(1-hp.sig_pi2)*ft,2,1)**2/(2*hp.sig_pi2) #constants eliminated anyway

    vect = torch.dot(log_prob,loss_enc) / hp.bt

    optim_enc.zero_grad()
    vect.backward() # minimize the quantity so on -ve sign
    optim_enc.step() # update parameters

def transmitter_step_sup(): # reinforced learning
    labels,ip = generate_input(amt=hp.bt)
    ft = encoder(ip)
    rec = add_channel_noise(ft)
    op = decoder(rec)

    loss_enc = loss_fn_dec(op,labels) # we use decoder loss since we want a scalar

    optim_enc.zero_grad()
    loss_enc.backward() # minimize the quantity so on -ve sign
    optim_enc.step() # update parameters

def receiver_step(): # supervised learning
    labels,ip = generate_input()
    with torch.no_grad():
        enc = encoder(ip)
    rec = add_channel_noise(enc)
    op = decoder(rec)

    loss_dec = loss_fn_dec(op,labels)
    optim_dec.zero_grad()
    loss_dec.backward() # compute gradients
    optim_dec.step() # update parameters


errs = []
#for epoch 0 - before any training
if hp.train:
    labels,ip = generate_input(amt=10*hp.br)
    enc = encoder(ip)
    enc = add_channel_noise(enc)
    op = decoder(enc)

    acc = accuracy(op,labels)
    loss = loss_fn_dec(op,labels)
    errs.append(error_rate(op,labels))
    log(0,loss,acc)

if hp.train:
    for _ in range(10*hp.n_batches_dec): # initial training
        receiver_step()
    for t in range(1,hp.n_epochs):
        for _ in range(hp.n_batches_enc):
            receiver_step()
            pass
        for _ in range(hp.n_batches_dec):
            if hp.sup:
                transmitter_step_sup()
            else:
                transmitter_step_rl()
            pass
        if hp.verbose >= 1:
            labels,ip = generate_input(amt=10*hp.br)
            enc = encoder(ip)
            enc = add_channel_noise(enc)
            op = decoder(enc)

            acc = accuracy(op,labels)
            loss = loss_fn_dec(op,labels)
            errs.append(error_rate(op,labels))
            log(t,loss,acc)
        if hp.verbose >= 2:
            print(next(encoder.parameters())[0][0:3])
            print(next(decoder.parameters())[0][0:3])

encoder.eval()
decoder.eval()
if hp.verbose >= 0:
    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc = add_channel_noise(enc)
    op = decoder(enc)
    loss = loss_fn_dec(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss with encoding:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )


if hp.plot and hp.train:
    try:
        os.makedirs('Training')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.figure(dpi=250)
    plt.plot(range(hp.n_epochs),errs,'-b')
    plt.title('Training profile of autoencoder ({0},{1})'.format(hp.n,hp.k))
    plt.xlabel('Training Epoch')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.savefig( join('Training','training_({0},{1})_{2}_{3}.png'.format(hp.n,hp.k,hp.SNR_dB,hp.sup)) )
    plt.show()

# saving if best performer
try:
    os.makedirs('Best')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

candidate = acc.cpu().detach().numpy()
try:
    best_acc = np.loadtxt(join('Best','acc({1},{2})_{0:.2f}_{3}.txt').format(hp.SNR,hp.n,hp.k,hp.sup))
except OSError:
    best_acc = 0

if candidate > best_acc:
    print('New best accuracy!')
    np.savetxt(join('Best','acc({1},{2})_{0:.2f}_{3}.txt').format(hp.SNR,hp.n,hp.k,hp.sup) , np.array([candidate]))
    copyfile('AWGN.py', join('Best','AWGN({1},{2})_{0}_{3}.py'.format(hp.SNR,hp.n,hp.k,hp.sup)) )
    torch.save(encoder, join('Best','encoder({1},{2})_{0}_{3}.pt'.format(hp.SNR,hp.n,hp.k,hp.sup)))
    torch.save(decoder, join('Best','decoder({1},{2})_{0}_{3}.pt'.format(hp.SNR,hp.n,hp.k,hp.sup)))
else:
    print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


if hp.inspect: # to view encodings, etc.
    for _ in range(1):
        labels,ip = generate_input(amt=1)
        print('Input:\t\t',ip.data.cpu().numpy()[0])
        enc = encoder(ip)
        print('Encoding:\t',enc.data.cpu().numpy()[0])
        enc = add_channel_noise(enc)
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
    plt.title('Constellation for AWGN ({0},{1})'.format(hp.n,hp.k))
    plt.savefig( join('Constellations','constellation_({0},{1}).png'.format(hp.n,hp.k)) )
    plt.show()


if hp.boundaries: # to (try to) visualize decision boundaries, etc.
    try:
        os.makedirs('Decision Boundaries')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ip = torch.eye(hp.M,device=device)
    enc = encoder(ip).cpu().detach().numpy()

    pca = PCA(2)

    enc_emb = pca.fit_transform(enc).T
    mean = enc_emb.mean(axis=1).reshape(2,1)
    std = enc_emb.std()
    enc_emb -= mean
    enc_emb /= std

    res = 2000

    xx = np.linspace(-2,2,res).reshape(-1,1)
    yy = np.linspace(-2,2,res)

    end = np.zeros((res,1))
    pts = []
    for y in yy:
        st = time()
        enc = np.concatenate((xx,end+y),axis=1)
        enc1 = pca.inverse_transform(enc)
        enc1 = torch.from_numpy( enc1 ).float().to(device)
        with torch.no_grad():
            op = torch.nn.Softmax(dim=1)(decoder(enc1))
        top2,_ = torch.topk(op,2,dim=1)
        top_diff = torch.abs(top2[:,0]-top2[:,1])

        for i,x in enumerate(top_diff):
            if x < 0.01:
                pts.append( [xx[i],y] )

    pts_emb = np.array(pts).T


    plt.figure(dpi=250)
    plt.scatter(pts_emb[0],pts_emb[1],c='r',s=3)
    plt.scatter(enc_emb[0],enc_emb[1])
    plt.title('Decision boundaries for AWGN ({0},{1})'.format(hp.n,hp.k))
    plt.savefig( join('Decision Boundaries','AWGN({0},{1}).png'.format(hp.n,hp.k)) )
    plt.show()



print( 'Total time taken:{0:.2f} seconds'.format(time()-start) )
