import numpy as np
import argparse
import torch
import pickle
import os

from shutil import copyfile
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=1) # number of epochs
parser.add_argument('--n_batches',type=int,default=200) # number of batches per epoch
parser.add_argument('--bs',type=int,default=128) # batch size
parser.add_argument('--sl',type=int,default=20) # signal length
parser.add_argument('--il',type=int,default=20) # intermediate length of network
parser.add_argument('--id',type=int,default=3) # intermediate depth
parser.add_argument('--el',type=int,default=20) # encoding length
parser.add_argument('--verbose',type=int,default=1) # verbosity
parser.add_argument('--lr',type=float,default=1e-4) # learning rate
parser.add_argument('--SNR',type=float,default=1) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.01) # signal to noise ratio

hyper = parser.parse_args()

# np.random.seed(101) # not for now

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# the network
if True:
    encoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.sl, hyper.il), torch.nn.ReLU(),

        * (torch.nn.Linear(hyper.il, hyper.il), torch.nn.ReLU())*(hyper.il-1),

        torch.nn.Linear(hyper.il, hyper.el), torch.nn.Sigmoid()
    )

    decoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.el, hyper.il), torch.nn.ReLU(),

        * (torch.nn.Linear(hyper.il, hyper.il), torch.nn.ReLU())*(hyper.il-1),

        torch.nn.Linear(hyper.il, hyper.sl), torch.nn.Sigmoid()
    )
else:
    encoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.sl, hyper.el), torch.nn.Sigmoid()
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(hyper.el, hyper.sl), torch.nn.Sigmoid()
    )

encoder.to(device)
decoder.to(device)
model = torch.nn.Sequential(encoder,decoder) # end to end model to be trained
model.to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr)

def log(t,loss,acc):
    print('{0}\tLoss:{1:.4e}\tAccuracy:{2:.2f}%'.format(t,loss.item(),acc))

def generate_input(amt=hyper.bs): # to generate inputs
    return torch.bernoulli( torch.ones(amt,hyper.sl)/2 ).to(device)

# something = torch.bernoulli( torch.ones(hyper.bs,hyper.sl)/2 ).to(device)
# def generate_input(): # to generate inputs
    # return something


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight,gain=1)
        # torch.nn.init.normal_(m.bias,std=hyper.init_std) # STD TERM PRETTY SENSITIVE
        torch.nn.init.eye_(m.weight)
        m.bias.data.fill_(0)
model.apply(weights_init)


def accuracy(y_true,y_pred):
    '''
    y_true must contain 1s and 0s
    y_pred would contain (0,1)
    '''
    # print(100 * ( 1 - torch.sum(torch.abs(y_true-torch.round(y_pred)))/y_true.numel() ))
    # exit()
    return 100 * ( 1 - torch.sum(torch.abs(y_true-torch.round(y_pred)))/y_true.numel() )

# learning rate scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 50, gamma=0.1, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.5, patience=20, verbose=True,
    threshold=1e-4, threshold_mode='rel',
    cooldown=5, min_lr=1e-6, eps=1e-08
    )

# print(next(model.parameters())[0])
# print(list(model.parameters())[0].grad[0])
# exit()

for t in range(hyper.n_epochs):
    for _ in range(hyper.n_batches):
        ip = generate_input()
        enc = encoder(ip)
        enc = enc + torch.randn_like(enc).to(device) / (2 * hyper.SNR)
        op = decoder(enc)

        loss = loss_fn(op,ip)

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.zero_grad()
    if hyper.verbose >= 1:
        acc = accuracy(ip,op)
        log(t,loss,acc)
    if hyper.verbose >= 2:
        print(next(model.parameters())[0][0:3])
    scheduler.step(loss.item())







if hyper.verbose >= 0:
    # Calculate loss with ANN encoding
    ip = generate_input(amt=1000)
    enc = encoder(ip)
    enc = enc + torch.randn_like(enc).to(device) / (2 * hyper.SNR)
    op = decoder(enc)
    loss = loss_fn(op,ip)
    acc = accuracy(ip,op)
    print( '\nLoss with encoding:{0:.4e}'.format( loss ) )
    print( 'Accuracy with encoding:{0:.2f}%\n\n'.format( acc ) )

    # Calculate loss without encoding
    ip = generate_input(amt=1000)
    op = ip + torch.randn_like(ip).to(device) / (2 * hyper.SNR)
    op = torch.clamp(op,min=0,max=1)
    pure_loss = loss_fn(op,ip)
    pure_acc = accuracy(ip,op)
    print( 'Loss without encoding:{0:.4e}'.format( pure_loss ) )
    print( 'Accuracy without encoding:{0:.2f}%\n\n'.format( pure_acc ) )

    # Calculate loss without encoding but double sending
    ip = generate_input(amt=1000)
    op = ip + torch.randn_like(ip).to(device) / (2 * hyper.SNR) / 2**0.5
    op = torch.clamp(op,min=0,max=1)
    double_loss = loss_fn(op,ip)
    double_acc = accuracy(ip,op)
    print( 'Loss with double encoding:{0:.4e}'.format( double_loss ) )
    print( 'Accuracy with double encoding:{0:.2f}%'.format( double_acc ) )




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













exit()
