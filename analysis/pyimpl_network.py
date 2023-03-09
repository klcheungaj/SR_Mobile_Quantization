import tensorflow as tf
from PIL import Image
import numpy as np
import pyimpl_conv
import tflite_get_tensor
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

img_size = 224
quan_bit = 16
max_exp = 1
min_exp = 1
scale = 3

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def quantized_mul(in_map, multiplier):
    global max_exp
    global min_exp
    mantissa, exponent = np.frexp(multiplier)
    temp_max = np.max(exponent)
    temp_min = np.min(exponent)
    if temp_max > max_exp:
        max_exp = temp_max
        print(f"maximum exponent is updadted: {max_exp}")
    if temp_min < min_exp:
        min_exp = temp_min
        print(f"minimum exponent is updadted: {min_exp}")
    quan_multiplier = np.rint(mantissa * (2**quan_bit)).astype('int64')
    quan_out_img = in_map * quan_multiplier

    ## round off for both neg & pos values
    neg_mask = quan_out_img < 0
    quan_out_img_abs = np.absolute(quan_out_img)
    quan_out_img_abs += 2**((quan_bit - 1 - exponent))
    quan_out_img_abs = np.right_shift(quan_out_img_abs, quan_bit-exponent)
    quan_out_img = quan_out_img_abs
    quan_out_img[neg_mask] *= -1
    # quan_out_img = quan_out_img * (2**(-quan_bit))

    real_out_img = multiplier*in_map

    # diff_quan_real = quan_out_img - real_out_img
    # max_idx = np.unravel_index(np.argmax(diff_quan_real, axis=None), diff_quan_real.shape)
    # min_idx = np.unravel_index(np.argmin(diff_quan_real, axis=None), diff_quan_real.shape)

    # print("max error: ", np.max(diff_quan_real), "|| quan: ", quan_out_img[max_idx], "|| real: ", real_out_img[max_idx], "|| input: ", in_map[max_idx], "|| scale: ", multiplier[max_idx[-1]], "|| index: ", max_idx)
    # print("min error: ", np.min(diff_quan_real), "|| quan: ", quan_out_img[min_idx], "|| real: ", real_out_img[min_idx], "|| input: ", in_map[min_idx], "|| scale: ", multiplier[min_idx[-1]], "|| index: ", min_idx)

    return quan_out_img
    # return real_out_img


img_path = 'data/0845.png'
img = Image.open(img_path)
input_data = np.array(img)
input_data = np.expand_dims(input_data, 0)
print(f"input_data shape: \n{input_data.shape}")

model_path = '../TFMODEL/base7_D4C28_bs16ps64_lr1e-3.tflite'
# Load the TFLite model and allocate tensors. View details
interpreter = tf.lite.Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
interpreter.resize_tensor_input(0, input_data.shape, strict=True)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()
operators_details = interpreter._get_ops_details()
print(f"input_details: \n{input_details[0]}")
# print(input_details)
# print("==================================================")
# print(output_details)
# print(interpreter.get_tensor_details())

## loading image
# img_path = "n02123394\\ILSVRC2012_val_00015898.JPEG"
# width, height = img.size
# if (width > height):
#     img_resize = img.resize((int(width/height*img_size), img_size))
# else:
#     img_resize = img.resize((img_size, int(height/width*img_size)))
# # img_resize.show()
# img_crop = crop_center(img_resize, img_size, img_size)
# input_data =  np.array(img_crop)
# input_data = tf.reshape(input_data, (1,224,224,3))
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()



## my test


def emulate_conv(in_map, in_idx, weight_idx, bias_idx, out_idx, stride, is_depthwise, isRelu):
    # in_map, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
    weights = interpreter.get_tensor(weight_idx)
    bias = interpreter.get_tensor(bias_idx)
    in_details = tensor_details[in_idx]
    weights_details = tensor_details[weight_idx]
    bias_details = tensor_details[bias_idx]
    out_details = tensor_details[out_idx]
    in_scale = in_details['quantization_parameters']['scales'][0]
    in_zp = in_details['quantization_parameters']['zero_points'][0]
    out_scale = out_details['quantization_parameters']['scales'][0]
    out_zp = out_details['quantization_parameters']['zero_points'][0]
    weight_scale = weights_details['quantization_parameters']['scales']
    weight_zp = weights_details['quantization_parameters']['zero_points'][0]
    in_scale = in_scale.astype('float64')
    out_scale = out_scale.astype('float64')
    weight_scale = weight_scale.astype('float64')

    in_map = in_map.astype('int64')
    in_map -= in_zp

    if (is_depthwise == True):
        print("depthwise")
        w_s = weights.shape
        weights = np.reshape(weights, (w_s[1],w_s[2],w_s[3]))
        out_img = pyimpl_conv.conv2d_depthwise(in_map, weights, stride, 'SAME')
    else:
        print("trad conv")
        weights = weights.transpose(1,2,3,0)
        out_img = pyimpl_conv.conv2d(in_map, weights, stride, 'SAME')
    out_img = np.add(out_img, bias)
    print("max bias: ", np.max(bias))
    print("min bias: ", np.min(bias))
    multiplier = in_scale * weight_scale / out_scale
    out_img_real = quantized_mul(in_map=out_img, multiplier=multiplier)

    out_img_real += out_zp
    out_img_real = np.rint(out_img_real)
    out_img_real = out_img_real.astype('int64')
    if (isRelu):
        out_img_real = np.maximum(out_img_real, out_zp)
    out_img_real = np.maximum(out_img_real, -128)
    out_img_real = np.minimum(out_img_real, 127)
    out_img_real = out_img_real.astype('int8')

    # # out_img_real = np.maximum(out_img_real, out2_zp)
    # # out_img_quan = quantizedMul(out_img, weight_scale)
    # # out_img_quan += out_zp
    # # out_img_quan = np.rint(out_img_quan)
    # # out_img_quan = out_img_quan.astype('int64')
    # # out_img_quan = np.maximum(out_img_quan, out_zp)

    # output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, out_idx)
    # # print(in_map[:,:,:,0])
    # # print(out_img_quan[:,:,:,0])
    # # print("=============================================")
    # print(out_img_real.shape, out_img_real[:,:,:,0])
    # print("=============================================")
    # print(output_true.shape, output_true[:,:,:,0])
    return out_img_real

def emulate_depth_to_space(input_data, input_idx, output_idx):
    # input_data = interpreter.get_tensor(input_idx)
    input_details = tensor_details[input_idx]
    output_details = tensor_details[output_idx]
    in_scale = input_details['quantization_parameters']['scales'][0]
    in_zp = input_details['quantization_parameters']['zero_points'][0]
    out_scale = output_details['quantization_parameters']['scales'][0]
    out_zp = output_details['quantization_parameters']['zero_points'][0]
    output_data = tf.nn.depth_to_space(input_data, scale).numpy()
    return output_data

def emulate_quantized_input(input_data, input_idx, output_idx):
    # input_data = interpreter.get_tensor(input_idx)
    input_details = tensor_details[input_idx]
    output_details = tensor_details[output_idx]
    in_scale = input_details['quantization_parameters']['scales'][0]
    in_zp = input_details['quantization_parameters']['zero_points'][0]
    out_scale = output_details['quantization_parameters']['scales'][0]
    out_zp = output_details['quantization_parameters']['zero_points'][0]
    input_scale = input_details['quantization_parameters']['scales'][0].astype('float64')
    output_scale = output_details['quantization_parameters']['scales'][0].astype('float64')
    multiplier = in_scale/out_scale
    input_data = input_data.astype('int32')
    # output = (in_scale/out_scale)*(input_data - in_zp) + out_zp
    input_data -= in_zp
    output = quantized_mul(in_map=input_data, multiplier=multiplier)
    output += out_zp
    # raw_in, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, 164)
    # quan_in, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, 0)
    # print(raw_in[:,:,:,0])
    # print(quan_in[:,:,:,0])
    print(f"data type: {output_details['dtype']}")
    if output_details['dtype'] == np.uint8:
        output = np.maximum(output, 0)
        output = np.minimum(output, 255)
        output = output.astype('uint8')
    elif output_details['dtype'] == np.int8:
        output = np.maximum(output, -128)
        output = np.minimum(output, 127)
        output = output.astype('int8')
    elif output_details['dtype'] == np.uint32:
        output = np.maximum(output, 0)
        output = np.minimum(output, 255)
        output = output.astype('uint32')
    elif output_details['dtype'] == np.int32:
        output = np.maximum(output, -128)
        output = np.minimum(output, 127)
        output = output.astype('int32')
    else:
        raise TypeError("quantized layer output type unexpected!")
    return output

def emulate_add_tensors(input1, in1_idx, input2, in2_idx, out_idx):
    # input1, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in1_idx)
    # input2, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in2_idx)
    input1_details = tensor_details[in1_idx]
    input2_details = tensor_details[in2_idx]
    output_details = tensor_details[out_idx]
    input1_scale = input1_details['quantization_parameters']['scales'][0]
    input1_zp = input1_details['quantization_parameters']['zero_points'][0]
    input2_scale = input2_details['quantization_parameters']['scales'][0]
    input2_zp = input2_details['quantization_parameters']['zero_points'][0]
    output_scale = output_details['quantization_parameters']['scales'][0]
    output_zp = output_details['quantization_parameters']['zero_points'][0]
    input1_scale = input1_scale.astype('float64')
    input2_scale = input2_scale.astype('float64')
    output_scale = output_scale.astype('float64')

    input1 = input1.astype('int64')
    input1 -= input1_zp
    input1_multiplier = input1_scale / output_scale
    input1 = quantized_mul(in_map=input1, multiplier=input1_multiplier)
    
    input2 = input2.astype('int64')
    input2 -= input2_zp
    input2_multiplier = input2_scale / output_scale
    input2 = quantized_mul(in_map=input2, multiplier=input2_multiplier)

    out = np.rint(input1 + input2).astype('int64') + output_zp

    # output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, out_idx)
    # print(out.shape, out[:,:,:,32])
    # print("=============================================")
    # print(output_true.shape, output_true[:,:,:,32])
    out = np.maximum(out, -128)
    out = np.minimum(out, 127)
    out = out.astype('int8')
    return out

def emulate_maxpool(in_map):
    # input1, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
    in_map = in_map.astype('int64')
    in_s = in_map.shape
    acc = np.sum(in_map, axis=(1,2)) / (in_s[1]*in_s[2])
    out = np.rint(acc).astype('int8')
    out = np.reshape(out, (in_s[0], 1, 1, in_s[3]))
    # output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, 160)
    # print(out.shape,out)
    # print("===============")
    # print(output_true.shape, output_true)
    return out

def emulate_reshape(in_map, shape_idx):
    shape = interpreter.get_tensor(shape_idx)
    output = np.reshape(in_map, (shape[0], shape[1]))
    return output

def emulate_fully_connected(in_map, in_idx, weight_idx, bias_idx, out_idx):
    # in_map, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
    weights = interpreter.get_tensor(weight_idx)
    bias = interpreter.get_tensor(bias_idx)
    in_details = tensor_details[in_idx]
    weights_details = tensor_details[weight_idx]
    bias_details = tensor_details[bias_idx]
    out_details = tensor_details[out_idx]
    in_scale = in_details['quantization_parameters']['scales'][0]
    in_zp = in_details['quantization_parameters']['zero_points'][0]
    out_scale = out_details['quantization_parameters']['scales'][0]
    out_zp = out_details['quantization_parameters']['zero_points'][0]
    weight_scale = weights_details['quantization_parameters']['scales']
    weight_zp = weights_details['quantization_parameters']['zero_points'][0]
    in_scale = in_scale.astype('float64')
    out_scale = out_scale.astype('float64')
    weight_scale = weight_scale.astype('float64')
    in_s = in_map.shape
    w_s = weights.shape
    b_s = bias.shape
    print("max bias: ", np.max(bias))
    print("min bias: ", np.min(bias))

    in_map = in_map.astype('int64')
    in_map -= in_zp
    in_map = np.reshape(in_map, (in_s[0], 1, 1, in_s[1]))
    weights = np.reshape(weights, (1, 1, w_s[0], w_s[1]))
    weights = weights.transpose(0,1,3,2)
    out_img = pyimpl_conv.conv2d(in_map, weights, 1, 'SAME')
    out_img = np.add(out_img, bias)
    multiplier = in_scale * weight_scale / out_scale
    out_img_real = quantized_mul(in_map=out_img, multiplier=multiplier)


    out_img_real += out_zp
    out_img_real = np.rint(out_img_real)
    out_img_real = out_img_real.astype('int64')
    out_img_real = np.maximum(out_img_real, out_zp)
    out_img_real = np.minimum(out_img_real, 127)
    out_img_real = out_img_real.astype('int8')
    temp_shape = out_img_real.shape
    out_img_real = np.reshape(out_img_real, (temp_shape[0], temp_shape[-1]))
    # output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, out_idx)
    # print(out_img_real.shape, out_img_real[:,:,:,76])
    # print("=============================================")
    # print(output_true.shape, output_true[:,76])
    # print("=============================================")
    # print(np.equal(out_img_real, output_true))
    return out_img_real

def emulate_quantized_softmax(in_map, in_idx, out_idx):
    max_uint8 = 255
    
    # in_map, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
    in_details = tensor_details[in_idx]
    out_details = tensor_details[out_idx]
    in_scale = in_details['quantization_parameters']['scales'][0]
    in_zp = in_details['quantization_parameters']['zero_points'][0]
    out_scale = out_details['quantization_parameters']['scales'][0]
    out_zp = out_details['quantization_parameters']['zero_points'][0]

    lut = []
    for val in range(max_uint8 + 1):
        temp = 2**quan_bit * math.exp(in_scale*(val-max_uint8))
        temp = np.rint(temp).astype('int64')
        lut.append(temp)
    in_map = in_map.astype('int64')     
    in_map += in_zp

    ## this is for 1 batch size
    acc = 0
    for idx in range(in_map.shape[1]):
        acc += lut[in_map[0,idx]]
    multiplier = np.rint(out_scale*(2**quan_bit)).astype('int64')
    acc *= multiplier
    acc = np.right_shift(acc, quan_bit)
    acc = int(np.rint(acc))
    out = np.zeros_like(in_map)
    for idx in range(out.shape[1]):
        out[0,idx] = np.rint((lut[in_map[0,idx]] / acc)) + out_zp
        # print(lut[in_map[0,idx]])
    # output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, out_idx)
    # print(np.equal(out, output_true))
    # print(out)
    out = out.astype('int8')
    return out

in_map_dict = {}
diff_dict = {}
def run_network(start_idx):
    in_map_temp = interpreter.get_tensor(start_idx)
    in_map_dict[start_idx] = in_map_temp
    for operator in operators_details:
        inputs = operator['inputs']
        outputs = operator['outputs']
        op_name = operator['op_name']
        index = operator['index']
        # if index > 1:
        #     break
        if op_name == 'QUANTIZE':
            in_idx = inputs[0]
            out_idx = outputs[0]
            in_map_temp = emulate_quantized_input(in_map_dict[in_idx], in_idx, out_idx)
            in_map_dict[out_idx] = in_map_temp
            # if (len(in_map_temp.shape) == 4):
            #     output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, out_idx)
            #     print(output_true[:,:,:,0])
            #     print("==============================")
            #     print(in_map_temp[:,:,:,0])
            #     print("==============================")
            # else:
            #     input_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
            #     print(input_true)
            #     print("==============================")
            #     print(in_map_dict[in_idx])
            #     print("==============================")
        elif op_name == 'RELU':
            in_idx = inputs[0]
            out_idx = outputs[0]
            in_map_temp = emulate_quantized_input(in_map_dict[in_idx], in_idx, out_idx)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'MINIMUM':
            in_idx = inputs[0]
            out_idx = outputs[0]
            in_map_temp = emulate_quantized_input(in_map_dict[in_idx], in_idx, out_idx)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'DEPTH_TO_SPACE':
            in_idx = inputs[0]
            out_idx = outputs[0]
            in_map_temp = emulate_depth_to_space(in_map_dict[in_idx], in_idx, out_idx)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'CONV_2D' or op_name == 'DEPTHWISE_CONV_2D':
            in_idx = inputs[0]
            weight_idx = inputs[1]
            bias_idx = inputs[2]
            out_idx = outputs[0]
            out_detail = tensor_details[out_idx]
            in_detail = tensor_details[in_idx]
            ## assume Width and Height have same stride
            stride = int(in_detail['shape'][1] / out_detail['shape'][1])
            out_name = tensor_details[out_idx]
            isRelu = True
            if "relu" not in out_name: 
                isRelu = False
            is_depthwise = True
            if op_name == 'CONV_2D':
                is_depthwise = False
            in_map_temp = emulate_conv(in_map=in_map_dict[in_idx], in_idx=in_idx, weight_idx=weight_idx, bias_idx=bias_idx, out_idx=out_idx, stride=stride, is_depthwise=is_depthwise, isRelu=isRelu)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'ADD':
            in1_idx = inputs[0]
            in2_idx = inputs[1]
            out_idx = outputs[0]
            
            ## for debug
            # in1_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in1_idx)
            # in2_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in2_idx)
            # print(in1_true[:,:,:,0])
            # print("==============================")
            # print(in_map_dict[in1_idx][:,:,:,0])
            # print("==============================")
            # print(in2_true[:,:,:,0])
            # print("==============================")
            # print(in_map_dict[in2_idx][:,:,:,0])
            # print("==============================")
            in_map_temp = emulate_add_tensors(input1=in_map_dict[in1_idx], in1_idx=in1_idx, input2=in_map_dict[in2_idx], in2_idx=in2_idx, out_idx=out_idx)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'AVERAGE_POOL_2D':
            in_idx = inputs[0]
            out_idx = outputs[0]

            ## for debug
            # output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
            # print(output_true[:,:,:,0])
            # print("==============================")
            # print(in_map_dict[in_idx][:,:,:,0])
            # print("==============================")

            in_map_temp = emulate_maxpool(in_map=in_map_dict[in_idx])
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'RESHAPE':
            in_idx = inputs[0]
            shape_idx = inputs[1]
            out_idx = outputs[0]
            in_map_temp = emulate_reshape(in_map=in_map_dict[in_idx], shape_idx=shape_idx)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'FULLY_CONNECTED':
            in_idx = inputs[0]
            weight_idx = inputs[1]
            bias_idx = inputs[2]
            out_idx = outputs[0]
            in_map_temp = emulate_fully_connected(in_map=in_map_dict[in_idx], in_idx=in_idx, weight_idx=weight_idx, bias_idx=bias_idx, out_idx=out_idx)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'SOFTMAX':
            in_idx = inputs[0]
            out_idx = outputs[0]
            in_map_temp = emulate_quantized_softmax(in_map=in_map_dict[in_idx], in_idx=in_idx, out_idx=out_idx)
            in_map_dict[out_idx] = in_map_temp
        elif op_name == 'CONCATENATION':
            in_idx = inputs[0]
            out_idx = outputs[0]
            ## FIXME: scale is hardcoded as 3
            in_map_dict[out_idx] = np.concatenate([in_map_dict[in_idx]]*(scale**2), axis=3)
            print(f"CONCATENATION input shape: {in_map_dict[in_idx].shape}, output shape: {in_map_dict[out_idx].shape}")
        else:
            print(f"Invalid operator name: {op_name}")
        print(f"op name: {op_name}")
        diff_acc = print_diff(in_map_dict[out_idx], out_idx=out_idx, isPrint=True, in_idx=in_idx)
        diff_dict[index] = diff_acc
    # output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, 165)
    # print(in_map_dict[165])
    # print("==============================")
    # print(output_true)
    # print("==============================")
    print(diff_dict)
    

def print_tensors(axis, idx):
    output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, idx)
    out_s = output_true.shape
    if len(out_s) > 3:
        print(output_true[:,:,:,axis])
        print("============================================")
        print(in_map_dict[idx][:,:,:,axis])
    else:
        print(output_true)
        print("============================================")
        print(in_map_dict[idx])

def print_diff(output_tensor, out_idx, isPrint=True, in_idx=0):
    output_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, out_idx)
    input_true, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
    out_s = output_true.shape
    diff = output_true.astype(np.int32) - output_tensor.astype(np.int32)
    print(f"maximum difference: {np.amax(diff)}")
    print(f"the maximum in expected data: {np.amax(output_true)}")
    print(f"the minimum in expected data: {np.amin(output_true)}")
    flat_max_idx = np.argmax(diff)
    max_idx = np.unravel_index(flat_max_idx, diff.shape)
    print(f"maximum error element in expected data: {output_true[max_idx]}")
    print(f"maximum error element in actual data: {output_tensor[max_idx]}")
    if output_true.shape == input_true.shape:
        print(f"maximum error element in true input data: {input_true[max_idx]}")
        # print(f"maximum error element in emulated input data: {in_map_dict[in_idx][max_idx]}")
    if isPrint:
        # with open("diff_out.txt", 'w') as file:
        #     file.write(f"diff output: \n{diff}")
        pass
    diff_acc = 0
    if len(out_s) > 3:
        diff_acc = np.count_nonzero(diff, axis=(0,1,2,3))
    else:
        diff_acc = np.count_nonzero(diff, axis=(0,1))
    return diff_acc

def run_layer(in_idx=27, out_idx=28):
    global model_path
    global input_data
    input_tensor, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, in_idx)
    expected_output_tensor, _ = tflite_get_tensor.get_intermediate_tensor(model_path, input_data, out_idx)
    output_tensor = emulate_quantized_input(input_tensor, in_idx, out_idx)
    with open("input.txt", 'w') as file:
        file.write(f"input output: \n{input_tensor}")
    with open("actual_output.txt", 'w') as file:
        file.write(f"actual output: \n{output_tensor}")
    with open("expected_output.txt", 'w') as file:
        file.write(f"expected output: \n{expected_output_tensor}")
    print_diff(output_tensor, out_idx=out_idx, isPrint=True, in_idx=in_idx)

run_network(0)
# print(tflite_get_tensor.get_intermediate_tensor(model_path, input_data, 0)[0][:,:,:,0])
"""
conclusion:
    qquantized conv: (input_scale * weights_scale / output_scale) * {(conv_acc[(input - input_zero_point)*(weights - weights_zero_point)] + bias} + output_zero_point
    quantized input: input - input_zero_point + output_zero_point
"""