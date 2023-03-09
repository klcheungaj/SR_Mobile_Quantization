import numpy as np
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tflite
import os
import tensorflow as tf

# def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
#   # Reads model_buffer as a proper flatbuffer file and gets the offset programatically
#   # It might be much more efficient if Model.subgraphs[0].outputs[] was set to a list of all the tensor indices.
#   fb_model_root = tflite.Model.GetRootAsModel(model_buffer, 0)
#   print(fb_model_root.Subgraphs(0))
#   output_tensor_index_offset = fb_model_root.Subgraphs(0) # Custom added function to return the file offset to this vector
#   # print("buffer_change_output_tensor_to. output_tensor_index_offset: ")
#   # print(output_tensor_index_offset)
#   # output_tensor_index_offset = 0x5ae07e0 # address offset specific to inception_v3.tflite
#   # output_tensor_index_offset = 0x16C5A5c # address offset specific to inception_v3_quant.tflite
#   # Flatbuffer scalars are stored in little-endian.
#   new_tensor_i_bytes = bytes([
#     new_tensor_i & 0x000000FF, \
#     (new_tensor_i & 0x0000FF00) >> 8, \
#     (new_tensor_i & 0x00FF0000) >> 16, \
#     (new_tensor_i & 0xFF000000) >> 24 \
#   ])
#   # Replace the 4 bytes corresponding to the first output tensor index
#   return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]

# def get_intermediate_tensor(model_file, input_data, tensor_idx):
#     # Read model into model_buffer for output tensor modification
#     model_buffer = None
#     with open(model_file, 'rb') as f:
#         model_buffer = f.read()

#     model_buffer = buffer_change_output_tensor_to(model_buffer, tensor_idx)
#     interpreter = interpreter_wrapper.Interpreter(model_content=model_buffer)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
    
#     # print(interpreter.get_output_details())
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     # Run inference on the input data up until the output tensor is calculated
#     interpreter.invoke()

#     # Get the tensor data
#     dets = interpreter._get_tensor_details(tensor_idx)
#     tens = interpreter.get_tensor(tensor_idx)

#     return tens, dets


def get_intermediate_tensor(model_file, input_data, tensor_idx):
    interpreter = tf.lite.Interpreter(model_path=model_file, experimental_preserve_all_tensors=True)
    interpreter.resize_tensor_input(0, input_data.shape, strict=True)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    operators_details = interpreter._get_ops_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Get the tensor data
    dets = interpreter._get_tensor_details(tensor_idx)
    tens = interpreter.get_tensor(tensor_idx)

    return tens, dets
