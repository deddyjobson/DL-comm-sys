import numpy as np
import tensorflow as tf
import os

from time import time

# Load TFLite model and allocate tensors.
for model in ['teacher','student']:
    interpreter = tf.lite.Interpreter(model_path=os.path.join('Best',"{0}.tflite".format(model)))
    # interpreter = tf.lite.Interpreter(model_path=os.path.join('Best',"student.tflite"))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time()
    interpreter.invoke() # run through model
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    print('Time taken by {1} :{0:.2f}ms'.format(10**3 * (time()-start) , model))
