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
from time import time
from tensorflow.keras.optimizers import Adam
from shutil import copyfile

#hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=300) # 0=> no training
parser.add_argument('--bs',type=int,default=1024) # batch size
parser.add_argument('--dr',type=float,default=0.5) # dropout rate
parser.add_argument('--tts',type=float,default=0.5) # train_test_split: train/test
parser.add_argument('--vb',type=int,default=1) # verbosity
parser.add_argument('--slc',type=int,default=1) # show loss curves
parser.add_argument('--pcm',type=int,default=1) # plot confusion matrix
parser.add_argument('--snr_cm',type=int,default=1) # show confusion matrix for various snr values
parser.add_argument('--pac',type=int,default=1) # plot accuracy curves
parser.add_argument('--psd',type=int,default=0) # plot sample data point
parser.add_argument('--lr',type=float,default=1e-3) # train_test_split: train/test

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

if hp.psd:
    idx = np.random.randint(0,X.shape[0])
    t = X[idx]
    t = np.linalg.norm(t,axis=0)
    plt.figure(dpi=300)
    plt.title('{0}—SNR_dB:{1}'.format(*lbl[idx]))
    plt.plot(t)
    plt.show()

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
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))


in_shp = list(X_train.shape[1:])
if hp.vb > 1:
    print(X_train.shape, in_shp)
    print('Classes:',classes)
classes = mods

# # Build the NN Model

# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

optim = Adam(hp.lr)

model = models.Sequential()
model.add(Reshape(in_shp+[1], input_shape=in_shp)) # tf format
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
model.add(Dropout(hp.dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(80, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
model.add(Dropout(hp.dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(hp.dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)

# # Train the Model

# perform training ...
#   - call the main training loop in tf.keras for our network+dataset
filename = os.path.join('Models','convmodrecnets_CNN2_{0}.wts.h5'.format(model.count_params()))

if hp.n_epochs > 0:
    history = model.fit(X_train,
        Y_train,
        batch_size=hp.bs,
        epochs=hp.n_epochs,
        verbose=hp.vb,
        validation_data=(X_test, Y_test), # WOT?
        callbacks = [
            keras.callbacks.ModelCheckpoint(filename, monitor='val_acc', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
        ])
# we re-load the best weights once training is finished
model.load_weights(filename)


# # Evaluate and Plot Model Performance

# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=hp.bs)
print('Test Evaluation:',score)


try:
    os.makedirs('Best')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    best_acc = keras.models.load_model( os.path.join('Best','model{0}.h5'.format(model.count_params())) ).evaluate(X_test, Y_test, verbose=0, batch_size=hp.bs)[1]
    print('Best Accuracy:',best_acc)
except OSError:
    best_acc = 0

candidate = score[1]
if candidate > best_acc:
    print('New best accuracy!')
    copyfile('RML2016.10a_VTCNN2_example.py', os.path.join('Best','bestRML2016.10a_{0}.py'.format(model.count_params())) )
    model.save( os.path.join('Best','model{0}.h5'.format(model.count_params())) )
else:
    print('Too bad, Best accuracy is {0:.2f}'.format(best_acc))

exit()

try:
    os.makedirs('Figures')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Show loss curves
if hp.slc and hp.n_epochs>0:
    plt.figure(dpi=300)
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig(os.path.join('Figures','training_performance.png'))
elif hp.slc:
    print('Need to also set --n_epochs to a value greater than 1 to plot loss curve')


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join('Figures','{0}.png'.format(title.lower().replace(' ','_'))))
    plt.close()


# Plot confusion matrix
if hp.pcm:
    test_Y_hat = model.predict(X_test, batch_size=hp.bs)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,X_test.shape[0]):
        j = list(Y_test[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plot_confusion_matrix(confnorm, labels=classes)

# Plot confusion matrix
if hp.snr_cm:
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
        plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("Overall Accuracy: ", cor / (cor+ncor))
        acc[snr] = 1.0*cor/(cor+ncor)


    # Save results to a pickle file for plotting later
    print(acc)
    with open(os.path.join('Data','results_cnn2_d0.5.dat'),'wb') as fd:
        pickle.dump( ("CNN2", 0.5, acc) , fd )


# Plot accuracy curve
if hp.pac and hp.snr_cm:
    if not hp.snr_cm:
        try:
            _,__,acc = pickle.load(os.path.join('Data','results_cnn2_d0.5.dat'),'rb')
        except:
            print('No presaved data available. Need to also set --snr_cm to 1 to plot accuracy curve')
            exit()
    plt.figure(dpi=300)
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
    plt.savefig(os.path.join('Figures','CNN2_classification_accuracy.png'))
