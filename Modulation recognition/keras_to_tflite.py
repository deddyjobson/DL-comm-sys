import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.keras.utils import CustomObjectScope

# convert teacher
converter = tf.lite.TFLiteConverter.from_keras_model_file("teacher_model9278475.h5")
tflite_model = converter.convert()
open("teacher.tflite", "wb").write(tflite_model)


custom_loss = K.categorical_crossentropy
# convert student
with CustomObjectScope({'custom_loss': custom_loss}):
    converter = tf.lite.TFLiteConverter.from_keras_model_file("student_model2830427.h5")
tflite_model = converter.convert()
open("student.tflite", "wb").write(tflite_model)
