# from tflite import Model
# import  numpy as np
# buf = open('../weights/efficientnet-lite0/efficientnet-lite0-int8.tflite', 'rb').read()


# model = Model.Model.GetRootAsModel(buf, 0)
# subgraph = model.Subgraphs(0)
# # Check tensor.Name() to find the tensor_idx you want
# tensor = subgraph.Tensors(52) 
# buffer_idx = tensor.Buffer()
# print('buffer_idx', buffer_idx)
# buffer = model.Buffers(buffer_idx)
# buffer_val = buffer.DataAsNumpy()
# print(buffer_val)
# # for idx in range (buffer.DataLength()):
# #     print(buffer.Data(idx))

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
# import netron 
import json

model_path = '../TFMODEL/base7_D4C28_bs16ps64_lr1e-3.tflite'

def find_memory_usage():
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    operators_details = interpreter._get_ops_details()
    tensor = interpreter.get_tensor(2)

    bias_mem_usage = 0
    weights_mem_usage = 0
    global_max = 0
    global_min = 0
    for idx in range(len(tensor_details)):
        if (idx > 0 and idx < 101):
            tensor = interpreter.get_tensor(idx)
            data_type = tensor.dtype
            if (data_type == np.int32):
                # temp_max = np.max(tensor)
                # if temp_max > global_max:
                #     print(temp_max)
                #     global_max = temp_max
                # temp_min = np.max(tensor)
                # if temp_min < global_min:
                #     global_min = temp_min

                mem = tensor.nbytes
                bias_mem_usage += mem
            else:   
                mem = tensor.nbytes
                weights_mem_usage += mem

    print("max: ", global_max, "min: ", global_min)
    print("weights memory usage: ", weights_mem_usage/1024, "Kb")
    print("bias memory usage: ", bias_mem_usage/1024, "Kb")

def print_tensor_buffer():
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    operators_details = interpreter._get_ops_details()
    # print(input_details)
    # print(output_details)
    print(operators_details)
    
    # print(type(tensor_details[103]['quantization_parameters']['scales'][0]))

    # for operator in operators_details:
    #     print(operator)

    # for tensor in tensor_details:
    #     print(tensor['index'], ' : ', tensor['dtype'], tensor['shape'])

print_tensor_buffer()
# netron.start('../weights/efficientnet-lite0/efficientnet-lite0-int8.tflite')
# print('end')


## conclusion
## tensor[164] is input, tensor[165] is output
## tensor[0] is quantized input``
## tensor[102:163] is hidden layer's feature map
## tensor[165] is output, tensor[163] is pre-quantized output