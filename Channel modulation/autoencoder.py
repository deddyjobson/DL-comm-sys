import numpy as np
import argparse
import torch
import pickle
import os
import errno

from shutil import copyfile
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=500) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=32) # batch size
parser.add_argument('--M',type=int,default=2) # signal length
parser.add_argument('--n',type=int,default=2) # encoding length
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-3) # learning rate
parser.add_argument('--SNR',type=float,default=1) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.01) # bias initialization
parser.add_argument('--write_out',type=int,default=0) # save performance
parser.add_argument('--e_prec',type=int,default=5) # precision of error

hyper = parser.parse_args()

# np.random.seed(101) # not for now

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SNR_dB = 10 * np.log10(hyper.SNR)

# the network
encoder = torch.nn.Sequential(
    torch.nn.Linear(hyper.M, hyper.M), torch.nn.ReLU(),
    torch.nn.Linear(hyper.M, hyper.n),
    torch.nn.BatchNorm1d(hyper.n) # contrains power of transmitter
)
decoder = torch.nn.Sequential(
    torch.nn.Linear(hyper.n, hyper.M), torch.nn.ReLU(),
    torch.nn.Linear(hyper.M, hyper.M)
    # , torch.nn.Softmax(dim=1) # automatically applied by loss function
)

encoder.to(device)
decoder.to(device)
model = torch.nn.Sequential(encoder,decoder) # end to end model to be trained
model.to(device)

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=hyper.init_std) # STD TERM PRETTY SENSITIVE
        # torch.nn.init.eye_(m.weight)
        # m.bias.data.fill_(0)
model.apply(weights_init)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=hyper.lr)

def log(t,loss,acc):
    print('{0}\tLoss:{1:.4e}\tAccuracy:{2:.2f}%'.format(t,loss.item(),acc))

def generate_input(amt=hyper.bs): # to generate inputs
    indices = torch.randint(low=0,high=hyper.M,size=(amt,))
    return indices,torch.eye(hyper.M)[indices]


def accuracy(y_true,y_pred):
    '''
    y_true must contain 1s and 0s
    y_pred would contain (0,1)
    '''
    # print(100 * ( 1 - torch.sum(torch.abs(y_true-torch.round(y_pred)))/y_true.numel() ))
    # exit()
    return 100 * ( 1 - torch.sum(torch.abs(y_true-torch.round(y_pred)))/y_true.numel() )

def accuracy(out, labels):
  outputs = torch.argmax(out, dim=1)
  return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()

def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.5, patience=10, verbose=hyper.verbose>0,
    threshold=1e-4, threshold_mode='rel',
    cooldown=5, min_lr=1e-7, eps=1e-08
    )

# print(next(model.parameters())[0])
# print(list(model.parameters())[0].grad[0])
# exit()



for t in range(hyper.n_epochs):
    for _ in range(hyper.n_batches):
        labels,ip = generate_input()
        enc = encoder(ip)
        enc = enc + torch.randn_like(enc).to(device) / hyper.SNR
        op = decoder(enc)

        # loss = loss_fn(labels,ip)
        loss = loss_fn(op,labels)

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.zero_grad()
    if hyper.verbose >= 1:
        acc = accuracy(op,labels)
        log(t,loss,acc)
    if hyper.verbose >= 2:
        print(next(model.parameters())[0][0:3])
    scheduler.step(loss.item())







if hyper.verbose >= 0:
    # Calculate loss with ANN encoding
    labels,ip = generate_input(amt=10**hyper.e_prec)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc).to(device) / hyper.SNR
    op = decoder(enc)
    loss = loss_fn(op,labels)
    acc = accuracy(op,labels)
    print( '\nLoss with encoding:{0:.4e}'.format( loss ) )
    print( 'Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )


# saving if best performer
try:
    os.makedirs('Best')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

candidate = acc.detach().numpy()
try:
    best_acc = np.loadtxt(join('Best','best_acc_{0:.2f}.txt').format(hyper.SNR))
except OSError:
    best_acc = 0

if candidate > best_acc:
    print('New best accuracy!')
    np.savetxt(join('Best','best_acc_{0:.2f}.txt').format(hyper.SNR) , np.array([candidate]))
    copyfile('autoencoder.py', join('Best','best_autoencoder_{0}.py'.format(hyper.SNR)) )
    torch.save(model.state_dict(), join('Best','best_model_{0}.pt'.format(hyper.SNR)))


if hyper.write_out:
    with open('results.csv','a+') as f:
        f.write( '{0},{1}\n'.format(SNR_dB,1-acc/100) )
    print('Added error values to results')










exit()
