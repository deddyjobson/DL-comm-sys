import numpy as np
import argparse
import torch
import os
import errno

from shutil import copyfile
from os.path import join
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=2000) # number of epochs
parser.add_argument('--n_batches',type=int,default=100) # number of batches per epoch
parser.add_argument('--bs',type=int,default=256) # batch size
parser.add_argument('--n',type=int,default=8) # encoding length
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-4) # learning rate
parser.add_argument('--teach_dB',type=float,default=4) # signal to noise ratio
parser.add_argument('--SNR_dB',type=float,default=8) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--T',type=float,default=50) # temperature
parser.add_argument('--rel',type=float,default=0.5) # relative weight of losses
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--train_teacher',type=int,default=0)
temp = 0
parser.add_argument('--best_student',type=int,default=temp) # whether to load best model

hp = parser.parse_args()
hp.M = 2 ** hp.k # number of messages
hp.SNR = 10 ** (hp.SNR_dB/10)
scaler = np.sqrt( hp.SNR * 2 * hp.k / hp.n )
start = time()


device = "cpu" # default
if hp.gpu and torch.cuda.is_available():
    device = "cuda:0"
elif hp.gpu:
    print('GPU not available.')

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=hp.init_std)

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

def channel_output(enc):
    return enc + torch.randn_like(enc,device=device) / scaler

loss_obj = torch.nn.CrossEntropyLoss()
def soft_cross_entropy(pred, target):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    softmax = torch.nn.Softmax(dim=1)
    return hp.T**2 * torch.mean(torch.sum(- softmax(target/hp.T) * logsoftmax(pred/hp.T), 1))

def loss_fn(op_stud,op_teach,labels):
    return (hp.rel * torch.nn.CrossEntropyLoss()(op_stud,labels)
            + soft_cross_entropy(op_stud,op_teach))

def teach_forward(ip):
    return teach_dec( channel_output( teach_enc(ip) ) )

def stud_forward(ip):
    return stud_dec( channel_output( stud_enc(ip) ) )


if not hp.train_teacher:
    try:
        teach_enc = torch.load( join('Best CM','teacher_encoder({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
        teach_dec = torch.load( join('Best CM','teacher_decoder({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
        teach_enc.to(device)
        teach_dec.to(device)
        model = torch.nn.Sequential(teach_enc,teach_dec) # end to end model
        model.to(device)
    except FileNotFoundError:
        print('Teacher model not found.')
        exit()
else:
    teach_enc = torch.nn.Sequential(
        torch.nn.Linear(hp.M, hp.M),torch.nn.ReLU(),
        torch.nn.Linear(hp.M, hp.n),
        torch.nn.BatchNorm1d(hp.n, affine=False) # contrains power of transmitter
    )
    teach_dec = torch.nn.Sequential(
        torch.nn.Linear(hp.n, hp.M), torch.nn.ReLU(),
        torch.nn.Linear(hp.M, hp.M)
    )

    teach_enc.to(device)
    teach_dec.to(device)
    model = torch.nn.Sequential(teach_enc,teach_dec) # end to end model
    model.to(device)

    model.apply(weights_init)


if hp.train_teacher:
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    for t in range(hp.n_epochs):
        for _ in range(hp.n_batches):
            labels,ip = generate_input()
            op = teach_forward(ip)
            loss = loss_obj(op,labels)
            optimizer.zero_grad()
            loss.backward() # compute gradients
            optimizer.step() # update parameters
        if hp.verbose >= 1:
            acc = accuracy(op,labels)
            log(t,loss,acc)

    labels,ip = generate_input(amt=10**hp.e_prec)
    op = teach_forward(ip)
    acc = accuracy(op,labels)
    print( 'Teacher Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )

    candidate = acc.cpu().detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best CM','best_teach_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best CM','best_teach_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k) , np.array([candidate]))
        copyfile('channel_modulation.py', join('Best CM','best_channel_modulation({1},{2})_{0}.py'.format(hp.SNR_dB,hp.n,hp.k)) )
        torch.save(teach_enc, join('Best CM','teacher_encoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(teach_dec, join('Best CM','teacher_decoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


teach_enc.eval()
teach_dec.eval()
# teacher model is ready for teaching

stud_enc = torch.nn.Sequential(
    torch.nn.Linear(hp.M, hp.n),
    torch.nn.BatchNorm1d(hp.n, affine=False) # contrains power of transmitter
)
stud_dec = torch.nn.Sequential(
    torch.nn.Linear(hp.n, hp.M), torch.nn.ReLU(),
)

stud_enc.to(device)
stud_dec.to(device)
model = torch.nn.Sequential(stud_enc,stud_dec) # end to end model
model.to(device)

model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

for t in range(hp.n_epochs):
    for _ in range(hp.n_batches):
        labels,ip = generate_input()
        op_teach = teach_forward(ip)
        op_stud = stud_forward(ip)

        loss = loss_fn(op_stud,op_teach,labels)
        optimizer.zero_grad()
        loss.backward() # compute gradients

        optimizer.step() # update parameters
    if hp.verbose >= 1:
        acc = accuracy(op_stud,labels)
        log(t,loss,acc)

model.eval()

if hp.verbose >= 0:
    labels,ip = generate_input(amt=10**hp.e_prec)
    op_teach = teach_forward(ip)
    op_stud = stud_forward(ip)
    acc = accuracy(op_stud,labels)
    teach_acc = accuracy(op_teach,labels)
    print( 'Teacher Accuracy:{0:.2f}%'.format( teach_acc ) )
    print( 'Student Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )


if not hp.best_student:
    try:
        os.makedirs('Best CM')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    candidate = acc.cpu().detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best CM','best_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best CM','best_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k) , np.array([candidate]))
        copyfile('channel_modulation.py', join('Best CM','best_channel_modulation({1},{2})_{0}.py'.format(hp.SNR_dB,hp.n,hp.k)) )
        torch.save(stud_enc, join('Best CM','student_encoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(stud_dec, join('Best CM','student_decoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))








exit()
