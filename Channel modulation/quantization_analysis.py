import numpy as np
import argparse
import torch
import pickle
import os
import errno
import matplotlib.pyplot as plt
import copy

from shutil import copyfile
from os.path import join
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=1000) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=64) # batch size
parser.add_argument('--n',type=int,default=7) # number of channels
parser.add_argument('--k',type=int,default=4) # number of bits
parser.add_argument('--depth',type=int,default=1) # number of hidden layers
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-3) # learning rate
parser.add_argument('--SNR',type=float,default=2.51188643150958) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--decay',type=float,default=0) # weight decay adam
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--load_best',type=int,default=0) # whether to load best model
parser.add_argument('--train',type=int,default=1) # whether to train
parser.add_argument('--inspect',type=int,default=1) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=1) # to visualize encodings
parser.add_argument('--boundaries',type=int,default=1) # to visualize encodings

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

def quantize_not_quantize(m,nbits=4):#quantize+dequantize
     model = copy.deepcopy(m) # let's not destroy the original
     with torch.no_grad():
         for p in model.parameters():
             mini = p.min()
             maxi = p.max()
             p -= mini
             p /= (maxi-mini)
             p *= 2**(nbits-1)
             torch.round(p,out=p)
             p /= 2**(nbits-1)
             p *= (maxi-mini)
             p += mini
         return model



encoder = torch.load( join('Best','best_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
decoder = torch.load( join('Best','best_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
encoder.to(device)
decoder.to(device)
model = torch.nn.Sequential(encoder,decoder) # end to end model
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr, weight_decay=hyper.decay)

encoder.eval()
decoder.eval()

q_encoder = quantize_not_quantize(encoder)
q_decoder = quantize_not_quantize(decoder)

print(encoder[0].weight[0])
print(q_encoder[0].weight[0])

#normal network
if hyper.verbose >= 0:
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss for Full Precision:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )

#quantized network
if hyper.verbose >= 0:
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = q_encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = q_decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss with quantization (8-bit):{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )


# now regu
encoder = torch.load( join('Best','best_regu_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
decoder = torch.load( join('Best','best_regu_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.n,hyper.k)) )
encoder.to(device)
decoder.to(device)
model = torch.nn.Sequential(encoder,decoder) # end to end model
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr, weight_decay=hyper.decay)

encoder.eval()
decoder.eval()

if hyper.verbose >= 0:
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss for Full Precision:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )
    
q_encoder = quantize_not_quantize(encoder)
q_decoder = quantize_not_quantize(decoder)
#quantized network
if hyper.verbose >= 0:
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = q_encoder(ip)
    enc = enc + torch.randn_like(enc,device=device) / scaler
    op = q_decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss regu quantization (8-bit):{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )

exit('Success')


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


if hyper.boundaries: # to (try to) visualize decision boundaries, etc.
    try:
        os.makedirs('Decision Boundaries')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ip = torch.eye(hyper.M,device=device)
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
    plt.title('Decision boundaries of autoencoder ({0},{1})'.format(hyper.n,hyper.k))
    plt.savefig( join('Decision Boundaries','({0},{1}).png'.format(hyper.n,hyper.k)) )
    plt.show()



print( 'Total time taken:{0:.2f} seconds'.format(time()-start) )
