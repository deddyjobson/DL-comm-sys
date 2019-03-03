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
parser.add_argument('--n_batches_t',type=int,default=100) # number of batches per epoch
parser.add_argument('--n_batches_r',type=int,default=100) # number of batches per epoch
parser.add_argument('--bt',type=int,default=256) # batch size
parser.add_argument('--br',type=int,default=256) # batch size
parser.add_argument('--n',type=int,default=8) # number of channels
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr_t',type=float,default=1e-4) # learning rate
parser.add_argument('--lr_r',type=float,default=3e-3) # learning rate
parser.add_argument('--SNR_dB',type=float,default=20) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--plot',type=int,default=1)
temp = 0
parser.add_argument('--load_best',type=int,default=temp) # whether to load best model
parser.add_argument('--train',type=int,default=1-temp) # whether to train
parser.add_argument('--inspect',type=int,default=temp) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=temp) # to visualize encodings

parser.add_argument('--sig_pi2',type=float,default=0.02) # to visualize encodings


hp = parser.parse_args()
hp.M = 2 ** hp.k # number of messages
hp.SNR = 10 ** (hp.SNR_dB/10)
scaler = np.sqrt( hp.SNR * hp.k/(2*hp.n) )
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
    coeff = torch.sqrt(torch.randn((enc.shape[0],1),device=device)**2+
            torch.randn((enc.shape[0],1),device=device)**2)/np.sqrt(2)
    noise = torch.randn_like(enc,device=device) / scaler
    return coeff*enc + noise


# make network
if hp.load_best:
    encoder = torch.load( join('Best','best_encoder_RBF_real_sup({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) )
    dec_params = torch.load( join('Best','best_decoder_RBF_real_sup({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)) ).to(device)
    def decoder(x):
        y = decoder.lin1(x)
        y = torch.tanh(y)
        y = decoder.lin2(y)
        y = x * torch.clamp(1/y,-20,20)
        # y = torch.clamp(x/y,-20,20)
        y = decoder.lin3(y)
        y = torch.nn.ReLU()(y)
        y = decoder.lin4(y)
        return y
    decoder.lin1 = dec_params[0]
    decoder.lin2 = dec_params[1]
    decoder.lin3 = dec_params[2]
    decoder.lin4 = dec_params[3]

    encoder.to(device)
    # decoder.to(device) # separate networks
else:
    encoder = torch.nn.Sequential(
        torch.nn.Linear(hp.M, hp.M),torch.nn.ELU(),
        torch.nn.Linear(hp.M, hp.n), #transmitting complex numbers
        torch.nn.BatchNorm1d(hp.n, affine=False) # contrains power of transmitter
    )
    def decoder(x):
        y = decoder.lin1(x)
        y = torch.tanh(y)
        y = decoder.lin2(y)
        # y = torch.clamp(x/y,-20,20)
        y = x * torch.clamp(1/y,-20,20)
        # y = (x / y)
        y = decoder.lin3(y)
        y = torch.nn.ReLU()(y)
        y = decoder.lin4(y)
        # y = torch.nn.Softmax(dim=1)(y)
        return y
    decoder.lin1 = torch.nn.Linear(hp.n, 10).to(device)
    decoder.lin2 = torch.nn.Linear(10, 1).to(device)
    decoder.lin3 = torch.nn.Linear(hp.n, hp.M).to(device)
    decoder.lin4 = torch.nn.Linear(hp.M, hp.M).to(device)

    dec_params = torch.nn.Sequential(
        decoder.lin1,
        decoder.lin2,
        decoder.lin3,
        decoder.lin4
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

def transmitter_step(): # SUPERVISED learning
    labels,ip = generate_input()
    enc = encoder(ip)
    rec = channel_output(enc)
    op = decoder(rec)

    loss_enc = loss_fn_dec(op,labels)
    optim_enc.zero_grad()
    loss_enc.backward()
    optim_enc.step()

def receiver_step(): # supervised learning
    labels,ip = generate_input()
    # with torch.no_grad():
    enc = encoder(ip)
    rec = channel_output(enc)
    op = decoder(rec)

    loss_dec = loss_fn_dec(op,labels)
    optim_dec.zero_grad()
    loss_dec.backward()
    for p in dec_params.parameters():
        if torch.isnan(p.grad).any():
            return 0
    optim_dec.step()
    return 1


errs = []
if hp.train:
    #for epoch 0 - before any training
    labels,ip = generate_input(amt=10*hp.br)
    enc = encoder(ip)
    enc = channel_output(enc)
    op = decoder(enc)

    acc = accuracy(op,labels)
    loss = loss_fn_dec(op,labels)
    errs.append(error_rate(op,labels))
    log(0,loss,acc)



model = torch.nn.Sequential(encoder,dec_params)
def train():
    top_acc = 0
    for t in range(1,hp.n_epochs):
        for _ in range(hp.n_batches_r):
            if not receiver_step():
                return
            # receiver_step()
            pass
        for _ in range(hp.n_batches_t):
            transmitter_step()
            pass
        if hp.verbose >= 1:
            labels,ip = generate_input(amt=10*hp.br)
            enc = encoder(ip)
            enc = channel_output(enc)
            op = decoder(enc)

            acc = accuracy(op,labels)
            loss = loss_fn_dec(op,labels)
            errs.append(error_rate(op,labels))
            log(t,loss,acc)
            if acc > top_acc and not torch.isnan(acc).any():
                top_acc = acc
                torch.save(model.state_dict(), 'temp.pt')

if hp.train:
    train()

model.load_state_dict(torch.load('temp.pt'))
os.remove('temp.pt')
model.eval()

if hp.verbose >= 0:
    labels,ip = generate_input(amt=10**hp.e_prec)
    enc = encoder(ip)
    enc = channel_output(enc)
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
    plt.title('Training profile of RBF (real) ({0},{1})'.format(hp.n,hp.k))
    plt.xlabel('Training Epoch')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.savefig( join('Training','RBF_real_sup_({0},{1})_{2}.png'.format(hp.n,hp.k,hp.SNR_dB)) )
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
        best_acc = np.loadtxt(join('Best','best_acc_RBF_real_sup({1},{2})_{0:.2f}.txt').format(hp.SNR,hp.n,hp.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best','best_acc_RBF_real_sup({1},{2})_{0:.2f}.txt').format(hp.SNR,hp.n,hp.k) , np.array([candidate]))
        copyfile('RBF_real_sup.py', join('Best','best_RBF_real_sup({1},{2})_{0}.py'.format(hp.SNR,hp.n,hp.k)) )
        torch.save(encoder, join('Best','best_encoder_RBF_real_sup({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
        torch.save(dec_params, join('Best','best_decoder_RBF_real_sup({1},{2})_{0}.pt'.format(hp.SNR,hp.n,hp.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


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
