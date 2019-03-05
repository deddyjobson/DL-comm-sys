import numpy as np
import argparse
import torch
import os
import errno
import matplotlib.pyplot as plt

from shutil import copyfile
from os.path import join
from sklearn.manifold import TSNE
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--n',type=int,default=7) # number of channels
parser.add_argument('--k',type=int,default=4) # number of bits
parser.add_argument('--teach_dB',type=float,default=4)
parser.add_argument('--SNR_dB',type=float,default=4) # signal to noise ratio
parser.add_argument('--low',type=float,default=-4)
parser.add_argument('--up',type=float,default=8)
parser.add_argument('--e_prec',type=int,default=6) # precision of error
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--calc_acc',type=int,default=0) # print test accuracy
parser.add_argument('--inspect',type=int,default=0) # to view network mappings
parser.add_argument('--constellation',type=int,default=1) # to visualize encodings
parser.add_argument('--err_profile',type=int,default=0) # error vs SNR
parser.add_argument('--kwrd',type=str,default='CM Chi2 MSE') # to visualize encodings


hp = parser.parse_args()
hp.M = 2 ** hp.k # number of messages
hp.SNR = 10 ** (hp.SNR_dB/10)
scaler = np.sqrt( hp.SNR * 2 * hp.k / hp.n )
device = "cpu" # default
if hp.gpu:
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        print('GPU not available.')


def generate_input(amt=1): # to generate inputs
    indices = torch.randint(low=0,high=hp.M,size=(amt,),device=device)
    return indices,torch.eye(hp.M,device=device)[indices]
def channel_output(enc):
    return enc + torch.randn_like(enc,device=device) / scaler
def teach_forward(ip):
    return teach_dec( channel_output( teach_enc(ip) ) )
def stud_forward(ip):
    return stud_dec( channel_output( stud_enc(ip) ) )
def stud_alone_forward(ip):
    return stud_alone_dec( channel_output( stud_alone_enc(ip) ) )
def accuracy(out, labels):
    outputs = torch.argmax(out, dim=1)
    return 100 * torch.sum(outputs==labels).to(dtype=torch.double)/labels.numel()
def error_rate(out,labels):
    return 1 - accuracy(out,labels)/100


teach_enc = torch.load( join('Best {0}'.format(hp.kwrd),'teacher_encoder({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
teach_dec = torch.load( join('Best {0}'.format(hp.kwrd),'teacher_decoder({1},{2})_{0}.pt'.format(hp.teach_dB,hp.n,hp.k)) )
stud_enc = torch.load( join('Best {0}'.format(hp.kwrd),'student_encoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)) )
stud_dec = torch.load( join('Best {0}'.format(hp.kwrd),'student_decoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)) )
stud_alone_enc = torch.load( join('Best {0}'.format(hp.kwrd),'stud_alone_encoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)) )
stud_alone_dec = torch.load( join('Best {0}'.format(hp.kwrd),'stud_alone_decoder({1},{2})_{0}.pt'.format(hp.SNR_dB,hp.n,hp.k)) )

teach_enc.to(device)
teach_dec.to(device)
stud_enc.to(device)
stud_dec.to(device)
stud_alone_enc.to(device)
stud_alone_dec.to(device)

teach_enc.eval()
teach_dec.eval()
stud_enc.eval()
stud_dec.eval()
stud_alone_enc.eval()
stud_alone_dec.eval()

if hp.calc_acc:
    labels,ip = generate_input(amt=10**hp.e_prec)
    op_teach = teach_forward(ip)
    op_stud = stud_forward(ip)
    op_stud_alone = stud_alone_forward(ip)
    acc = accuracy(op_stud,labels)
    acc_alone = accuracy(op_stud_alone,labels)
    teach_acc = accuracy(op_teach,labels)
    print( 'Teacher Accuracy:{0:.2f}%'.format( teach_acc ) )
    print( 'Student Accuracy:{0:.2f}%'.format( acc ) )
    print( 'Student Alone   :{0:.2f}%'.format( acc_alone ) )
    print( 'Error rate:{0:.2e}\n\n'.format( 1-acc/100 ) )

if hp.inspect: # to view encodings, etc.
    labels,ip = generate_input(amt=1)
    print('Input:\t\t',ip.data.cpu().numpy()[0])
    enc = stud_enc(ip)
    print('Encoding:\t',enc.data.cpu().numpy()[0])
    enc = channel_output(enc)
    print('Channel:\t',enc.data.cpu().numpy()[0])
    op = stud_dec(enc)
    print('Output:\t\t',torch.softmax(op,dim=1).data.cpu().numpy()[0])

if hp.constellation: # to visualize encodings, etc.
    try:
        os.makedirs('Constellations')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    ip = torch.eye(hp.M,device=device)

    plt.figure(dpi=250)
    plt.grid()

    enc = stud_enc(ip).cpu().detach().numpy()
    enc_emb = TSNE().fit_transform(enc).T
    enc_emb -= enc_emb.mean(axis=1).reshape(2,1)
    enc_emb /= enc_emb.std()
    plt.scatter(enc_emb[0],enc_emb[1],label='Student')

    enc = stud_alone_enc(ip).cpu().detach().numpy()
    enc_emb = TSNE().fit_transform(enc).T
    enc_emb -= enc_emb.mean(axis=1).reshape(2,1)
    enc_emb /= enc_emb.std()
    plt.scatter(enc_emb[0],enc_emb[1],label='Student alone')

    enc = teach_enc(ip).cpu().detach().numpy()
    enc_emb = TSNE().fit_transform(enc).T
    enc_emb -= enc_emb.mean(axis=1).reshape(2,1)
    enc_emb /= enc_emb.std()
    plt.scatter(enc_emb[0],enc_emb[1],label='Teacher')

    plt.title('Constellation for {2} ({0},{1})'.format(hp.n,hp.k,hp.kwrd))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.legend()
    plt.savefig( join('Constellations','{2}_({0},{1}).png'.format(hp.n,hp.k,hp.kwrd)) )
    plt.show()

if hp.err_profile:
    try:
        os.makedirs('Error Profile')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    low = hp.low
    up = hp.up
    num_pts = 2*(up-low)+1

    snr_dBs = np.linspace( low,up,num_pts )
    snrs = 10 ** (snr_dBs/10)

    errs = np.zeros((3,num_pts))

    for i,snr in enumerate(snrs):
        print(num_pts - i) # countdown
        scaler = np.sqrt( snr * 2 * hp.k / hp.n )

        labels,ip = generate_input(amt=10**hp.e_prec)
        op_teach = teach_forward(ip)
        op_stud = stud_forward(ip)
        op_stud_alone = stud_alone_forward(ip)

        errs[0,i] = error_rate(op_teach,labels)
        errs[1,i] = error_rate(op_stud,labels)
        errs[2,i] = error_rate(op_stud_alone,labels)


    xx = snr_dBs
    yy = errs + 1 / 10**hp.e_prec # to protect against anomalous behaviour

    plt.figure(dpi=250)
    axes = plt.gca()
    axes.set_xlim([low,up])
    axes.set_ylim([1e-5,1e0])
    plt.grid()
    plt.semilogy(xx,yy[0],'-ob',label='Teacher')
    plt.semilogy(xx,yy[2],'-or',label='Student Alone')
    plt.semilogy(xx,yy[1],'-og',label='Student')
    plt.title('Error profile for {2} ({0},{1})'.format(hp.n,hp.k,hp.kwrd))
    plt.xlabel('$E_b/N_0$ [dB]')
    plt.ylabel('Block Error Rate')
    plt.legend()
    plt.savefig( join('Error Profile','{2}_({0},{1}).png'.format(hp.n,hp.k,hp.kwrd)) )
    plt.show()
