import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.keras.utils import CustomObjectScope
from os.path import join

QUANTIZE = True
custom_loss = K.categorical_crossentropy


# convert teacher
converter = tf.lite.TFLiteConverter.from_keras_model_file(join("Best","teacher_model9278475.h5"))
tflite_model = converter.convert()
open(join("Best","teacher.tflite"), "wb").write(tflite_model)

# convert student
with CustomObjectScope({'custom_loss': custom_loss}):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(join("Best","student_model2830427.h5"))
tflite_model = converter.convert()
open(join("Best","student.tflite"), "wb").write(tflite_model)

# convert student
if QUANTIZE:
    with CustomObjectScope({'custom_loss': custom_loss}):
        converter = tf.lite.TFLiteConverter.from_keras_model_file(join("Best","student_model2830427.h5"))
        converter.post_training_quantize=True
tflite_model = converter.convert()
open(join("Best","student8.tflite"), "wb").write(tflite_model)
