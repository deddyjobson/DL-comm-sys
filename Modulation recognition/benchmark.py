import numpy as np
import tensorflow as tf
import os
import pickle

from time import time


input_shape = (1,2,128)
num_pts = 1000

# loading dataset
Xd = pickle.load(open(os.path.join("Data","RML2016.10a_dict.pkl"),'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
# lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
X = np.vstack(X)[:num_pts].reshape((-1,*input_shape)).astype(np.float32)

print('Measuring inference time for {0} samples...'.format(X.shape[0]))
# Load TFLite model and allocate tensors.
for model in ['teacher','student','student8']:
    total_time = 0
    interpreter = tf.lite.Interpreter(model_path=os.path.join('Best',"{0}.tflite".format(model)))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    for i in range(X.shape[0]):
        interpreter.set_tensor(input_details[0]['index'], X[i])

        start = time()
        interpreter.invoke() # run through model
        total_time += time() - start
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    print('Time taken by {1} :{0:.2f}s'.format(total_time , model))
