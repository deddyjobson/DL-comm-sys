import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt

from time import time

bs = 1
num_pts = 110000
input_shape = [bs,2,128]

# loading dataset
Xd = pickle.load(open(os.path.join("Data","RML2016.10a_dict.pkl"),'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):
            lbl.append((mod,snr))

classes = mods

X = np.vstack(X)

np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
# test_idx = np.random.choice(list(set(range(0,n_examples))-set(train_idx)), size = num_pts)
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))


X_test = X_test.reshape([num_pts]+input_shape[1:]).astype(np.float32)

del Xd,X

plt.figure(dpi=300)
print('Measuring inference time for {0} samples...'.format(X_test.shape[0]))
for model,plot_label in zip(['student','student8'] , ['32-bit','8-bit']):
    Y_pred = []
    total_time = 0
    interpreter = tf.lite.Interpreter(model_path=os.path.join('Best',"{0}.tflite".format(model)))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(0,X_test.shape[0],bs):
        interpreter.set_tensor(input_details[0]['index'], X_test[i:i+bs])

        start = time()
        interpreter.invoke() # run through model
        total_time += time() - start

        output_data = interpreter.get_tensor(output_details[0]['index'])
        Y_pred.extend(np.argmax(output_data,axis=1))

    Y_pred = np.array(Y_pred)
    # pickle.dump(Y_pred,open(os.path.join('Models','y_preds.pkl'),'wb')) # for safety
    accuracy = np.sum(Y_pred==Y_test) / Y_test.shape[0]
    print('Accuracy: {0}'.format(accuracy))
    print('Time taken by {1} :{0:.2f}s'.format(total_time , model))

    acc = {}
    for snr in snrs:
        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        # test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

        # estimate classes
        test_Y_i_hat = Y_pred[np.where(np.array(test_SNRs)==snr)]
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = int(test_Y_i[i])
            # j = list(test_Y_i[i,:]).index(1)
            k = int(test_Y_i_hat[i])
            # k = int(np.argmax(test_Y_i_hat[i,:]))
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
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)),'-o',label=plot_label)

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.legend()
plt.savefig(os.path.join('Figures','q_classification_accuracy.png'))
