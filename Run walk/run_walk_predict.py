import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tflite_runtime.interpreter import Interpreter
#import tensorflow as tf

# Load the TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="rw_seq_model.tflite")
interpreter = Interpreter(model_path="rw_seq_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)

input_data = np.array([[0.6028,-1.0966,-0.3046,0.1674,-0.5065,1.0156 ]], dtype=np.float32)
print(input_data.shape)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Walk status:" + str(output_data[0][0]))


input_data_run = np.array([[-1.0739,0.0033,0.1052,2.1021,-0.9863,0.2939]],dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data_run)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Run status:"+ str(output_data[0][0]))


#print("Predict for run [{:.6%}]".(output_data))