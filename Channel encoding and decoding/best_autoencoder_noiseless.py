import numpy as np
import argparse
import torch
import pickle

from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=100) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=16) # batch size
parser.add_argument('--sl',type=int,default=20) # signal length
parser.add_argument('--il',type=int,default=30) # intermediate length of network
parser.add_argument('--id',type=int,default=3) # intermediate depth
parser.add_argument('--el',type=int,default=40) # encoding length
parser.add_argument('--lr',type=float,default=1e-4) # learning rate

hyper = parser.parse_args()

# np.random.seed(101) # not for now

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

model = torch.nn.Sequential(encoder,decoder) # end to end model to be trained
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr)

def log(t,loss):
    print('{0}\t\t Loss:{1:.4e}'.format(t,loss.item()))

def generate_input(): # to generate inputs
    return torch.bernoulli( torch.ones(hyper.bs,hyper.sl)/2 )

something = torch.bernoulli( torch.ones(hyper.bs,hyper.sl)/2 )
def generate_input(): # to generate inputs
    return something


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=0.01)
        # torch.nn.init.eye(m.weight)
        # m.bias.data.fill_(0)
model.apply(weights_init)


# learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)

# before training
ip = generate_input()
op = model(ip)
loss = loss_fn(op,ip)
log(-1,loss)
loss.backward()

# print(next(model.parameters())[0])
# print(list(model.parameters())[0].grad[0])
# exit()

for t in range(hyper.n_epochs):
    for _ in range(hyper.n_batches):
        # scheduler.step()
        ip = generate_input()
        op = model(ip)
        loss = loss_fn(op,ip)

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.zero_grad()
    log(t,loss)
    # print(next(model.parameters())[0])







candidate = loss.item()
try:
    with open('best_loss.pkl','rb') as f:
        best_loss = pickle.load(f)
except FileNotFoundError:
    best_loss = 1000

if candidate < best_loss:
    print('New best loss!')
    with open('best_loss.pkl','wb') as f:
        pickle.dump(candidate,f)
    copyfile('autoencoder.py', 'best_autoencoder.py')













exit()
