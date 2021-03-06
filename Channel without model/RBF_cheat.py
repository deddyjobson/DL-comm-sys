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
parser.add_argument('--n_epochs',type=int,default=1000) # number of epochs
parser.add_argument('--n_batches_t',type=int,default=100) # number of batches per epoch
parser.add_argument('--n_batches_r',type=int,default=100) # number of batches per epoch
parser.add_argument('--bt',type=int,default=256) # batch size
parser.add_argument('--br',type=int,default=256) # batch size
parser.add_argument('--n',type=int,default=8) # number of channels
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr_t',type=float,default=1e-4) # learning rate
parser.add_argument('--lr_r',type=float,default=1e-4) # learning rate
parser.add_argument('--SNR_dB',type=float,default=20) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--plot',type=int,default=0)
temp = 0
parser.add_argument('--load_best',type=int,default=temp) # whether to load best model
parser.add_argument('--train',type=int,default=1-temp) # whether to train
parser.add_argument('--inspect',type=int,default=temp) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=temp) # to visualize encodings

parser.add_argument('--sig_pi2',type=float,default=0.02) # to visualize encodings


hp = parser.parse_args()
hp.M = 2 ** hp.k # number of messages
hp.SNR = 10 ** (hp.SNR_dB/10)
# scaler = np.sqrt( 2 * 2 * hp.SNR )
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

def channel_output(enc): # enc.size() = [-1,2*hp.n]
    # coeff = torch.randn((enc.shape[0],1),device=device)
    coeff = torch.sqrt(torch.randn((enc.shape[0],1),device=device)**2+
            torch.randn((enc.shape[0],1),device=device)**2)/2
    # coeff = torch.randn((1,),device=device)
    noise = torch.randn_like(enc,device=device) / scaler
    return coeff*enc + noise, coeff


# make network
if hp.load_best:
    encoder = torch.load( join('Best','best_encoder_RBF_cheat({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) )
    dec_params = torch.load( join('Best','best_decoder_RBF_cheat({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) ).to(device)
    def decoder(x,coeffs):
        y = x / coeffs
        y = decoder.lin1(y)
        y = torch.nn.ReLU()(y)
        y = decoder.lin2(y)
        return y
    decoder.lin1 = dec_params[0]
    decoder.lin2 = dec_params[1]

    encoder.to(device)
    # decoder.to(device) # separate networks
else:
    encoder = torch.nn.Sequential(
        torch.nn.Linear(hp.M, hp.M),torch.nn.ELU(),
        torch.nn.Linear(hp.M, hp.n), #transmitting complex numbers
        torch.nn.BatchNorm1d(hp.n, affine=False) # contrains power of transmitter
    )
    def decoder(x,coeffs):
        y = torch.clamp(x / coeffs, -20,20) # to protect gradients
        y = decoder.lin1(y)
        y = torch.nn.ReLU()(y)
        y = decoder.lin2(y)
        # y = torch.nn.Softmax(dim=1)(y)
        return y
    decoder.lin1 = torch.nn.Linear(hp.n, hp.M).to(device)
    decoder.lin2 = torch.nn.Linear(hp.M, hp.M).to(device)

    dec_params = torch.nn.Sequential(
        decoder.lin1,
        decoder.lin2
    )

    encoder.to(device)

    encoder.apply(weights_init)
    dec_params.apply(weights_init)

loss_fn_enc = torch.nn.CrossEntropyLoss(reduction='none')
loss_fn_dec = torch.nn.CrossEntropyLoss()

optim_enc = torch.optim.Adam(encoder.parameters(), lr=hp.lr_t, weight_decay=hp.decay)
optim_dec = torch.optim.Adam(dec_params.parameters(), lr=hp.lr_r, weight_decay=hp.decay)

def transmit_sample(enc): # to transmit the encoded signals
    return np.sqrt(1-hp.sig_pi2)*enc + np.sqrt(hp.sig_pi2)*torch.randn_like(enc,device=device)

def normalize(enc):
    return enc / torch.norm(enc,dim=1).reshape(-1,1) * np.sqrt(hp.n) # normalization

def transmitter_step(): # reinforced learning
    optim_enc.zero_grad()
    labels,ip = generate_input(amt=hp.bt)
    ft = encoder(ip)
    with torch.no_grad():
        xp = transmit_sample(ft)

    rec ,coeffs= channel_output(xp)
    op = decoder(rec,coeffs)

    with torch.no_grad(): # to get reference for judging punishments and rewards
        reward = loss_fn_enc(op,labels)

    log_prob = - torch.norm(xp - np.sqrt(1-hp.sig_pi2)*ft,2,1)**2 / (2*hp.sig_pi2) #constants eliminated anyway
    vect = torch.dot(log_prob,reward) / hp.bt
    vect.backward() # compute gradients
    optim_enc.step() # update parameters


def transmitter_step(): # supervised learning
    labels,ip = generate_input()
    enc = encoder(ip)
    rec ,coeffs= channel_output(enc)
    op = decoder(rec,coeffs)

    loss_enc = loss_fn_dec(op,labels)
    optim_enc.zero_grad()
    loss_enc.backward()
    optim_enc.step()

def receiver_step(): # supervised learning
    labels,ip = generate_input()
    # with torch.no_grad():
    enc = encoder(ip)
    rec,coeffs = channel_output(enc)
    op = decoder(rec,coeffs)

    loss_dec = loss_fn_dec(op,labels)
    optim_dec.zero_grad()
    loss_dec.backward()
    optim_dec.step()


errs = []
if hp.train:
    #for epoch 0 - before any training
    labels,ip = generate_input(amt=10*hp.br)
    enc = encoder(ip)
    enc,coeffs = channel_output(enc)
    op = decoder(enc,coeffs)

    acc = accuracy(op,labels)
    loss = loss_fn_dec(op,labels)
    errs.append(error_rate(op,labels))
    log(0,loss,acc)

if hp.train:
    for t in range(1,hp.n_epochs):
        for _ in range(hp.n_batches_r):
            receiver_step()
            pass
        for _ in range(hp.n_batches_t):
            transmitter_step()
            pass
        if hp.verbose >= 1:
            labels,ip = generate_input(amt=10*hp.br)
            enc = encoder(ip)
            enc,coeffs = channel_output(enc)
            op = decoder(enc,coeffs)

            acc = accuracy(op,labels)
            loss = loss_fn_dec(op,labels)
            errs.append(error_rate(op,labels))
            log(t,loss,acc)


encoder.eval()
dec_params.eval()
if hp.verbose >= 0:
    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc,coeffs = channel_output(enc)
    op = decoder(enc,coeffs)
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
    plt.title('Training profile of RBF (cheat) ({0},{1})'.format(hp.n,hp.k))
    plt.xlabel('Training Epoch')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.savefig( join('Training','RBF_cheat_({0},{1})_{2}.png'.format(hp.n,hp.k,hp.SNR_dB)) )
    plt.show()


# saving if best performer
if not hp.load_best:
    try:
        os.makedirs('Best')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    candidate = acc.cpu().detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best','best_acc_RBF_cheat({1},{2})_{0:.2f}.txt').format(hp.SNR,hp.n,hp.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best','best_acc_RBF_cheat({1},{2})_{0:.2f}.txt').format(hp.SNR,hp.n,hp.k) , np.array([candidate]))
        copyfile('RBF_cheat.py', join('Best','best_RBF_cheat({1},{2})_{0}.py'.format(hp.SNR,hp.n,hp.k)) )
        torch.save(encoder, join('Best','best_encoder_RBF_cheat({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
        torch.save(dec_params, join('Best','best_decoder_RBF_cheat({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


exit()


if hp.inspect: # to view encodings, etc.
    for _ in range(1):
        labels,ip = generate_input(amt=1)
        print('Input:\t\t',ip.data.cpu().numpy()[0])
        enc = encoder(ip)
        print('Encoding:\t',enc.data.cpu().numpy()[0])
        enc = channel_output(enc)
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
    plt.title('Constellation for RBF ({0},{1})'.format(hp.n,hp.k))
    plt.savefig( join('Constellations','RBF({0},{1}).png'.format(hp.n,hp.k)) )
    plt.show()

print( 'Total time taken:{0:.2f} seconds'.format(time()-start) )
