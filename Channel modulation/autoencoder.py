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


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=500) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=32) # batch size
parser.add_argument('--M',type=int,default=8) # signal length
parser.add_argument('--n',type=int,default=8) # encoding length
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-3) # learning rate
parser.add_argument('--SNR',type=float,default=1) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.01) # bias initialization
parser.add_argument('--write_out',type=int,default=0) # save performance
parser.add_argument('--e_prec',type=int,default=5) # precision of error
temp = 0
parser.add_argument('--load_best',type=int,default=temp) # whether to load best model
parser.add_argument('--train',type=int,default=1-temp) # whether to train
parser.add_argument('--inspect',type=int,default=temp) # to make sure things are fine
parser.add_argument('--constellation',type=int,default=temp) # for analysis

hyper = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SNR_dB = 10 * np.log10(hyper.SNR)

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=hyper.init_std) # STD TERM PRETTY SENSITIVE

def log(t,loss,acc):
    print('{0}\tLoss:{1:.4e}\tAccuracy:{2:.2f}%'.format(t,loss.item(),acc))

def generate_input(amt=hyper.bs): # to generate inputs
    indices = torch.randint(low=0,high=hyper.M,size=(amt,))
    return indices,torch.eye(hyper.M)[indices]

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100


if hyper.load_best:
    encoder = torch.load( join('Best','best_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.M,hyper.n)) )
    decoder = torch.load( join('Best','best_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.M,hyper.n)) )
    encoder.to(device)
    decoder.to(device)
    model = torch.nn.Sequential(encoder,decoder) # end to end model
    model.to(device)
else: # create the network
    encoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.M, hyper.M), torch.nn.ReLU(),
        torch.nn.Linear(hyper.M, hyper.n),
        torch.nn.BatchNorm1d(hyper.n, affine=False) # contrains power of transmitter
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.n, hyper.M), torch.nn.ReLU(),
        torch.nn.Linear(hyper.M, hyper.M)
    )

    encoder.to(device)
    decoder.to(device)
    model = torch.nn.Sequential(encoder,decoder) # end to end model
    model.to(device)

    model.apply(weights_init)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=hyper.lr)


if hyper.train:
    for t in range(hyper.n_epochs):
        for _ in range(hyper.n_batches):
            labels,ip = generate_input()
            enc = encoder(ip)
            enc = enc + torch.randn_like(enc).to(device) / np.sqrt(hyper.SNR)
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
    enc = enc + torch.randn_like(enc).to(device) / np.sqrt(hyper.SNR)
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

    candidate = acc.detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best','best_acc({1},{2})_{0:.2f}.txt').format(hyper.SNR,hyper.M,hyper.n))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best','best_acc({1},{2})_{0:.2f}.txt').format(hyper.SNR,hyper.M,hyper.n) , np.array([candidate]))
        copyfile('autoencoder.py', join('Best','best_autoencoder({1},{2})_{0}.py'.format(hyper.SNR,hyper.M,hyper.n)) )
        torch.save(encoder, join('Best','best_encoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.M,hyper.n)))
        torch.save(decoder, join('Best','best_decoder({1},{2})_{0}.pt'.format(hyper.SNR,hyper.M,hyper.n)))


if hyper.write_out:
    with open('results.csv','a+') as f:
        f.write( '{0},{1}\n'.format(SNR_dB,1-acc/100) )
    print('Added error values to results')


if hyper.inspect: # to view encodings, etc.
    for _ in range(1):
        labels,ip = generate_input(amt=1)
        print('Input:\t\t',ip.data.numpy()[0])
        enc = encoder(ip)
        print('Encoding:\t',enc.data.numpy()[0])
        enc = enc + torch.randn_like(enc).to(device) / np.sqrt(hyper.SNR)
        print('Channel:\t',enc.data.numpy()[0])
        op = decoder(enc)
        print('Output:\t\t',torch.softmax(op,dim=1).data.numpy()[0])


if hyper.constellation: # to visualize encodings, etc.
    try:
        os.makedirs('Constellations')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    labels,ip = torch.arange(hyper.M,dtype=torch.int), torch.eye(hyper.M)
    enc = encoder(ip).detach().numpy()

    enc_emb = TSNE().fit_transform(enc).T
    enc_emb -= enc_emb.mean(axis=1).reshape(2,1)
    enc_emb /= enc_emb.std()

    plt.figure(dpi=250)
    plt.grid()
    plt.scatter(enc_emb[0],enc_emb[1])
    plt.title('Constellation of autoencoder ({0},{1})'.format(hyper.M,hyper.n))
    plt.savefig( join('Constellations','constellation_({0},{1}).png'.format(hyper.M,hyper.n)) )
    plt.show()






exit()
