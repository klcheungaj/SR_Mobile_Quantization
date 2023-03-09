
import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import matplotlib.pyplot as plt

model_path = "../TFMODEL/base7_D4C28_bs16ps64_lr1e-3.tflite"
image_dir = "./data"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.resize_tensor_input(0, [1, 452, 680, 3], strict=True)
interpreter.allocate_tensors()
print(interpreter.get_tensor_details())

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_data_type = input_details[0]["dtype"]
print(str(input_details))
print(str(output_details))
print(str(input_shape))
print(str(input_data_type))

file_list = os.listdir(image_dir)
infer_time = 0
start_time = time.time()
end_time = 0
for file in file_list:
    full_path = os.path.join(image_dir, file)

    input_image = np.array(Image.open(full_path), dtype=input_data_type)
    input_image_resize = np.reshape(input_image, (1, 452, 680, 3))
    input_image_raw = np.zeros_like(input_image_resize)
    print(f"height: {input_image_resize.shape[1]}")
    print(f"width: {input_image_resize.shape[2]}")
    for h in range(input_image_resize.shape[1]):
        for w in range(input_image_resize.shape[2]):
            if h%2==0 and w%2==0:   ## red
                # input_image_raw[0][h][w][0] = 255
                input_image_raw[0][h][w][0] = input_image_resize[0][h][w][0]
                input_image_raw[0][h][w+1][0] = input_image_resize[0][h][w][0]
                input_image_raw[0][h+1][w][0] = input_image_resize[0][h][w][0]
                input_image_raw[0][h+1][w+1][0] = input_image_resize[0][h][w][0]
            elif h%2==1 and w%2==1:   ## blue
                # input_image_raw[0][h][w][2] = 255
                input_image_raw[0][h][w][2] = input_image_resize[0][h][w][2]
                input_image_raw[0][h][w-1][2] = input_image_resize[0][h][w][2]
                input_image_raw[0][h-1][w][2] = input_image_resize[0][h][w][2]
                input_image_raw[0][h-1][w-1][2] = input_image_resize[0][h][w][2]
            else:
                # input_image_raw[0][h][w][1] = 255
                input_image_raw[0][h][w][1] = input_image_resize[0][h][w][1]
                if h%2==0:    
                    input_image_raw[0][h][w-1][1] = input_image_resize[0][h][w][1]
                else:
                    input_image_raw[0][h][w+1][1] = input_image_resize[0][h][w][1]
    print(input_image_raw)
    print(input_image_raw.shape)
    print(type(input_image_raw[0][0][0][0]))
    im = Image.fromarray(input_image_raw[0])
    im.save("test2_raw.jpeg")


    interpreter.set_tensor(input_details[0]['index'], input_image_raw)
    print("invoking")
    interpreter.invoke()
    infer_time = time.time() - start_time
    print(f"completed. Infer time: {infer_time}")
    output_image = interpreter.get_tensor(output_details[0]['index'])
    print(f"output shape: {output_image.shape}; type: {output_image.dtype}")
    output_image_resize = np.reshape(output_image, (1356, 2040, 3))
    im = Image.fromarray(output_image_resize)
    im.save("test2.jpeg")