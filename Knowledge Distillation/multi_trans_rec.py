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
parser.add_argument('--bs',type=int,default=128) # batch size
parser.add_argument('--n',type=int,default=4) # number of channels
parser.add_argument('--k',type=int,default=8) # number of bits
parser.add_argument('--verbose',type=int,default=1) # verbosity: higher the verbosier
parser.add_argument('--lr',type=float,default=1e-3) # learning rate
parser.add_argument('--teach_dB',type=float,default=10) # signal to noise ratio
parser.add_argument('--SNR_dB',type=float,default=10) # signal to noise ratio
parser.add_argument('--init_std',type=float,default=0.1) # bias initialization
parser.add_argument('--e_prec',type=int,default=5) # precision of error
parser.add_argument('--T',type=float,default=20) # temperature
parser.add_argument('--rel',type=float,default=0.5) # relative weight of losses
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--tt',type=int,default=0) # train teacher
parser.add_argument('--tsa',type=int,default=1) # train student alone

hp = parser.parse_args()
hp.M = 2 ** hp.k # number of messages
hp.SNR = 10 ** (hp.SNR_dB/10)
scaler = np.sqrt( hp.SNR * hp.k / hp.n )
start = time()


device = "cpu" # default
if hp.gpu and torch.cuda.is_available():
    device = "cuda:0"
elif hp.gpu:
    print('GPU not available.')

try:
    os.makedirs('Best MTR')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        torch.nn.init.normal_(m.bias,std=hp.init_std)

def log(t,acc):
    log.mov_acc = log.mov_mom * log.mov_acc + acc
    den = 1 - log.mov_mom
    temp = log.mov_acc * (1 - log.mov_mom) / (1 - log.mov_mom**(t+1))
    print('{0}\tAcc:{1:.2f}%\tMoving:{2:.2f}%'.format(t,acc,temp))
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

def teach_forward(ip1,ip2):
    tot = teach_enc1(ip1)+teach_enc2(ip2)
    return teach_dec1( channel_output(tot) ), teach_dec2( channel_output(tot) )

def stud_forward(ip1,ip2):
    tot = stud_enc1(ip1)+stud_enc2(ip2)
    return stud_dec1( channel_output(tot) ), stud_dec2( channel_output(tot) )


if not hp.tt:
    try:
        teach_enc1 = torch.load( join('Best MTR','teacher_encoder1({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
        teach_dec1 = torch.load( join('Best MTR','teacher_decoder1({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
        teach_enc2 = torch.load( join('Best MTR','teacher_encoder2({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
        teach_dec2 = torch.load( join('Best MTR','teacher_decoder2({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
        teach_enc1.to(device)
        teach_dec1.to(device)
        teach_enc2.to(device)
        teach_dec2.to(device)
    except FileNotFoundError:
        print('Teacher model not found.')
        exit()
else:
    teach_enc1 = torch.nn.Sequential(
        torch.nn.Linear(hp.M, hp.M),torch.nn.ReLU(),
        torch.nn.Linear(hp.M, 2*hp.n),
        torch.nn.BatchNorm1d(2*hp.n, affine=False) # contrains power of transmitter
    )
    teach_dec1 = torch.nn.Sequential(
        torch.nn.Linear(2*hp.n, hp.M), torch.nn.ReLU(),
        torch.nn.Linear(hp.M, hp.M)
    )
    teach_enc2 = torch.nn.Sequential(
        torch.nn.Linear(hp.M, hp.M),torch.nn.ReLU(),
        torch.nn.Linear(hp.M, 2*hp.n),
        torch.nn.BatchNorm1d(2*hp.n, affine=False) # contrains power of transmitter
    )
    teach_dec2 = torch.nn.Sequential(
        torch.nn.Linear(2*hp.n, hp.M), torch.nn.ReLU(),
        torch.nn.Linear(hp.M, hp.M)
    )
    teach_enc1.to(device)
    teach_dec1.to(device)
    teach_enc2.to(device)
    teach_dec2.to(device)
    teach_enc1.apply(weights_init)
    teach_dec1.apply(weights_init)
    teach_enc2.apply(weights_init)
    teach_dec2.apply(weights_init)

print('Training Teacher...')
if hp.tt:
    params = list(teach_enc1.parameters())+list(teach_enc2.parameters())+list(teach_dec1.parameters())+list(teach_dec2.parameters())
    optim = torch.optim.Adam(params, lr=hp.lr)
    alpha = 0.5 # for first epoch
    for t in range(hp.n_epochs):
        for _ in range(hp.n_batches):
            labels1,ip1 = generate_input()
            labels2,ip2 = generate_input()
            op1,op2 = teach_forward(ip1,ip2)

            loss1 = loss_obj(op1,labels1)
            loss2 = loss_obj(op2,labels2)
            net_loss = alpha*loss1 + (1-alpha)*loss2
            with torch.no_grad():
                alpha = loss1/(loss1+loss2)

            optim.zero_grad()
            net_loss.backward()
            optim.step() # update parameters
        if hp.verbose >= 1:
            acc1 = accuracy(op1,labels1)
            acc2 = accuracy(op1,labels1)
            log(t,(acc1+acc2)/2)

labels1,ip1 = generate_input(amt=10**hp.e_prec)
labels2,ip2 = generate_input(amt=10**hp.e_prec)
op1,op2 = teach_forward(ip1,ip2)
losses = [loss_obj(op1,labels1),loss_obj(op2,labels2)]
accs = [accuracy(op1,labels1),accuracy(op2,labels2)]

for i in range(2):
    print('\nFor system {0},'.format(i))
    print( 'Loss with encoding:{0:.4e}'.format( losses[i] ) )
    print( 'Accuracy:{0:.2f}%'.format( accs[i] ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-accs[i]/100 ) )

acc = (accs[0] + accs[1])/2
teach_acc = acc

if hp.tt:
    candidate = acc.cpu().detach().numpy()
    try:
        best_acc = np.loadtxt(join('Best MTR','best_teach_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best MTR','best_teach_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k) , np.array([candidate]))
        copyfile('multi_trans_rec.py', join('Best MTR','best_multi_trans_rec({1},{2})_{0}.py'.format(hp.SNR_dB,hp.n,hp.k)) )
        torch.save(teach_enc1, join('Best MTR','teacher_encoder1({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(teach_dec1, join('Best MTR','teacher_decoder1({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(teach_enc2, join('Best MTR','teacher_encoder2({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(teach_dec2, join('Best MTR','teacher_decoder2({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


teach_enc1.eval()
teach_dec1.eval()
teach_enc2.eval()
teach_dec2.eval()
# teacher model is ready for teaching

stud_enc1 = torch.nn.Sequential(
    torch.nn.Linear(hp.M, 2*hp.n),
    torch.nn.BatchNorm1d(2*hp.n, affine=False) # contrains power of transmitter
)
stud_dec1 = torch.nn.Sequential(
    torch.nn.Linear(2*hp.n, hp.M)
)
stud_enc2 = torch.nn.Sequential(
    torch.nn.Linear(hp.M, 2*hp.n),
    torch.nn.BatchNorm1d(2*hp.n, affine=False) # contrains power of transmitter
)
stud_dec2 = torch.nn.Sequential(
    torch.nn.Linear(2*hp.n, hp.M)
)
stud_enc1.to(device)
stud_dec1.to(device)
stud_enc2.to(device)
stud_dec2.to(device)
stud_enc1.apply(weights_init)
stud_dec1.apply(weights_init)
stud_enc2.apply(weights_init)
stud_dec2.apply(weights_init)

params = list(stud_enc1.parameters())+list(stud_enc2.parameters())+list(stud_dec1.parameters())+list(stud_dec2.parameters())
optim = torch.optim.Adam(params, lr=hp.lr)

# train student alone
log.mov_acc = 0
if hp.tsa:
    print('Training Student Alone...')
    alpha = 0.5 # for first epoch
    for t in range(hp.n_epochs):
        for _ in range(hp.n_batches):
            labels1,ip1 = generate_input()
            labels2,ip2 = generate_input()
            op1,op2 = stud_forward(ip1,ip2)
            loss1 = loss_obj(op1,labels1)
            loss2 = loss_obj(op2,labels2)
            net_loss = alpha*loss1 + (1-alpha)*loss2
            with torch.no_grad():
                alpha = loss1/(loss1+loss2)
            optim.zero_grad()
            net_loss.backward()
            optim.step() # update parameters
        if hp.verbose >= 1:
            acc1 = accuracy(op1,labels1)
            acc2 = accuracy(op1,labels1)
            log(t,(acc1+acc2)/2)

    labels1,ip1 = generate_input(amt=10**hp.e_prec)
    labels2,ip2 = generate_input(amt=10**hp.e_prec)
    op1,op2 = stud_forward(ip1,ip2)
    losses = [loss_obj(op1,labels1),loss_obj(op2,labels2)]
    accs = [accuracy(op1,labels1),accuracy(op2,labels2)]

    for i in range(2):
        print('\nFor system {0},'.format(i))
        print( 'Loss with encoding:{0:.4e}'.format( losses[i] ) )
        print( 'Accuracy:{0:.2f}%'.format( accs[i] ) )
        print( 'Error rate:{0:.2e}\n\n'.format( 1-accs[i]/100 ) )

    acc = (accs[0] + accs[1])/2
    candidate = acc.cpu().detach().numpy()
    stud_acc_alone = acc
    try:
        best_acc = np.loadtxt(join('Best MTR','best_stud_alone_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k))
    except OSError:
        best_acc = 0

    if candidate > best_acc:
        print('New best accuracy!')
        np.savetxt(join('Best MTR','best_stud_alone_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k) , np.array([candidate]))
        torch.save(stud_enc1, join('Best MTR','stud_alone_encoder1({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(stud_dec1, join('Best MTR','stud_alone_decoder1({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(stud_enc2, join('Best MTR','stud_alone_encoder2({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
        torch.save(stud_dec2, join('Best MTR','stud_alone_decoder2({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
    else:
        print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))


stud_enc1.train()
stud_dec1.train()
stud_enc2.train()
stud_dec2.train()
stud_enc1.apply(weights_init)
stud_dec1.apply(weights_init)
stud_enc2.apply(weights_init)
stud_dec2.apply(weights_init)
# train student with the help of teacher
print('Training Student with help of Teacher...')
log.mov_acc = 0
alpha = 0.5 # for first epoch
for t in range(hp.n_epochs):
    for _ in range(hp.n_batches):
        labels1,ip1 = generate_input()
        labels2,ip2 = generate_input()
        op1,op2 = stud_forward(ip1,ip2)
        loss1 = loss_obj(op1,labels1)
        loss2 = loss_obj(op2,labels2)
        net_loss = alpha*loss1 + (1-alpha)*loss2
        with torch.no_grad():
            alpha = loss1/(loss1+loss2)
        optim.zero_grad()
        net_loss.backward()
        optim.step() # update parameters
    if hp.verbose >= 1:
        acc1 = accuracy(op1,labels1)
        acc2 = accuracy(op1,labels1)
        log(t,(acc1+acc2)/2)

stud_enc1.eval()
stud_dec1.eval()
stud_enc2.eval()
stud_dec2.eval()

labels1,ip1 = generate_input(amt=10**hp.e_prec)
labels2,ip2 = generate_input(amt=10**hp.e_prec)
op1,op2 = stud_forward(ip1,ip2)
losses = [loss_obj(op1,labels1),loss_obj(op2,labels2)]
accs = [accuracy(op1,labels1),accuracy(op2,labels2)]

for i in range(2):
    print('\nFor system {0},'.format(i))
    print( 'Loss with encoding:{0:.4e}'.format( losses[i] ) )
    print( 'Accuracy:{0:.2f}%'.format( accs[i] ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-accs[i]/100 ) )
stud_acc = (accs[0] + accs[1])/2

if hp.verbose >= 0:
    print( '\n\nTeacher Accuracy:{0:.2f}%'.format( teach_acc ) )
    print( 'Student Accuracy:{0:.2f}%'.format( stud_acc ) )
    if hp.tsa:
        print( 'Student Alone   :{0:.2f}%'.format( stud_acc_alone ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )

candidate = stud_acc.cpu().detach().numpy()
try:
    best_acc = np.loadtxt(join('Best MTR','best_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k))
except OSError:
    best_acc = 0

if candidate > best_acc:
    print('New best accuracy!')
    np.savetxt(join('Best MTR','best_acc({1},{2})_{0:.2f}.txt').format(hp.SNR_dB,hp.n,hp.k) , np.array([candidate]))
    copyfile('channel_modulation.py', join('Best MTR','best_channel_modulation({1},{2})_{0}.py'.format(hp.SNR_dB,hp.n,hp.k)) )
    torch.save(stud_enc1, join('Best MTR','student_encoder1({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
    torch.save(stud_enc2, join('Best MTR','student_encoder2({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
    torch.save(stud_dec1, join('Best MTR','student_decoder1({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
    torch.save(stud_dec2, join('Best MTR','student_decoder2({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)))
else:
    print('Too bad, Best accuracy is {0:.2f}%'.format(best_acc))








exit()
