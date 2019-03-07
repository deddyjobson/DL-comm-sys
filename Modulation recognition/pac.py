#!/usr/bin/env python3
# coding: utf-8

# ## Modulation Recognition Example: RML2016.10a Dataset + VT-CNN2 Mod-Rec Network
#
# This work is copyright DeepSig Inc. 2017.
# It is provided open source under the Create Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) Licence
# https://creativecommons.org/licenses/by-nc/4.0/
#
# Use of this work, or derivitives inspired by this work is permitted for non-commercial usage only and with explicit citaiton of this original work.
#
# A more detailed description of this work can be found at
# https://arxiv.org/abs/1602.04105
#
# A more detailed description of the RML2016.10a dataset can be found at
# http://pubs.gnuradio.org/index.php/grcon/article/view/11
#
# Citation of this work is required in derivative works:
#
# ```
# @article{convnetmodrec,
#   title={Convolutional Radio Modulation Recognition Networks},
#   author={O'Shea, Timothy J and Corgan, Johnathan and Clancy, T. Charles},
#   journal={arXiv preprint arXiv:1602.04105},
#   year={2016}
# }
# @article{rml_datasets,
#   title={Radio Machine Learning Dataset Generation with GNU Radio},
#   author={O'Shea, Timothy J and West, Nathan},
#   journal={Proceedings of the 6th GNU Radio Conference},
#   year={2016}
# }
# ```
#
# The RML2016.10a dataset is used for this work (https://radioml.com/datasets/)
#


# Import all the things we need ---
import os
import numpy as np
from tensorflow import keras
import pickle
import sys
import random
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.models as models
import argparse
import errno

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.regularizers import *
from tensorflow.keras import backend as K
from time import time
from tensorflow.keras.optimizers import Adam
from shutil import copyfile
from os.path import join

#hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=300) # 0=> no training
parser.add_argument('--bs',type=int,default=1024) # batch size
parser.add_argument('--dr',type=float,default=0.5) # dropout rate
parser.add_argument('--tts',type=float,default=0.5) # train_test_split: train/test
parser.add_argument('--vb',type=int,default=1) # verbosity
parser.add_argument('--snr_cm',type=int,default=1) # show confusion matrix for various snr values
parser.add_argument('--lr',type=float,default=1e-3) # train_test_split: train/test
parser.add_argument('--pc_teach',type=int,default=9278475) # parameter count of teacher model
parser.add_argument('--pc_stud',type=int,default=2830427) # parameter count of student model
parser.add_argument('--pc_stud_alone',type=int,default=2830427) # parameter count of student model
# parser.add_argument('--rel',type=float,default=0.5) # train_test_split: train/test
parser.add_argument('--T',type=float,default=1) # temperature for distillation

hp = parser.parse_args()

# # Dataset setup

# Load the dataset ...
start = time()
#  You will need to seperately download or generate this file
Xd = pickle.load(open(os.path.join("Data","RML2016.10a_dict.pkl"),'rb'))

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):
            lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * hp.tts)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))



in_shp = list(X_train.shape[1:])
if hp.vb > 1:
    print(X_train.shape, in_shp)
    print('Classes:',classes)
classes = mods

def custom_loss(target,output):
    output **= (1/hp.T)
    output /= K.reshape(K.sum(output,axis=1) , (-1,1))
    return (hp.T)**2 * K.categorical_crossentropy(target,output)


teach = models.load_model( os.path.join('Best','teacher_model{0}.h5'.format(hp.pc_teach)) )

stud = models.load_model( os.path.join('Best','student_model{0}.h5'.format(hp.pc_stud)), custom_objects={'custom_loss': custom_loss} )

stud_alone = models.load_model( os.path.join('Best','model{0}.h5'.format(hp.pc_stud_alone)) )
# best_acc = teacher.evaluate(X_test, Y_test, verbose=0, batch_size=hp.bs)[1]
# print('Best Teacher Accuracy:',best_acc)


# # Evaluate and Plot Model Performance

# Show simple version of performance
# score = model.evaluate(X_test, Y_test, verbose=0, batch_size=hp.bs)
# print('Test Evaluation:',score)
try:
    os.makedirs('Figures')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

plt.figure(dpi=300)
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy on RadioML 2016.10 Alpha (a)")
for model,clr,label in zip([teach,stud_alone,stud],['b','r','g'],['teacher','student alone','student']):
    acc = {}
    for snr in snrs:
        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        # test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("Overall Accuracy: ", cor / (cor+ncor))
        acc[snr] = 1.0*cor/(cor+ncor)

    # Save results to a pickle file for plotting later
    print(acc)
    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)),color=clr,label=label)

plt.legend()
plt.savefig(os.path.join('Figures','all_classification_accuracy.png'))
