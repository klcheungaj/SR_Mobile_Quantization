import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add, SeparableConv2D, Layer, MaxPooling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal, he_normal
import tensorflow.keras.backend as K
import numpy as np
from keras_flops import get_flops

class DoubleConv(Layer):    
    def __init__(self, out_ch):        
        super(DoubleConv, self).__init__()
        self.out_ch = out_ch
        self.conv1 = SeparableConv2D(self.out_ch, 3, padding='same', activation='relu', depthwise_initializer=he_normal(), pointwise_initializer=he_normal(), bias_initializer='zeros')
        self.conv2 = SeparableConv2D(self.out_ch, 3, padding='same', activation='relu', depthwise_initializer=he_normal(), pointwise_initializer=he_normal(), bias_initializer='zeros')
    
    def call(self, x):        
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def Unet_like(scale=2, in_channels=3, num_fea=14, m=3, out_channels=3):
    inp = Input(shape=(1920, 1080, 3)) 
    featureMap = {}
    ## downsampling
    x = inp
    for i in range(m):
        x = DoubleConv(num_fea)(x)
        featureMap[i] = x
        num_fea = num_fea * 2
        x = MaxPooling2D(pool_size=2)(x)
    
    x = DoubleConv(num_fea)(x)

    ## upsampling
    for i in reversed(range(m)):
        num_fea = num_fea // 2
        x = Conv2DTranspose(num_fea, 3, strides=2, padding='same', activation='relu', kernel_initializer=he_normal(), bias_initializer='zeros')(x)
        x = Concatenate(axis=2)([x, featureMap[i]])
        x = DoubleConv(num_fea)(x)

    out = SeparableConv2D(out_channels, 3, padding='same', depthwise_initializer=he_normal(), pointwise_initializer=he_normal(), bias_initializer='zeros')(x)

    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(out)
    
    model = Model(inputs=inp, outputs=out)
    print(model.summary())
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    return model

if __name__ == '__main__':
    model = Unet_like()
    print('Params: [{:.2f}]K'.format(model.count_params()/1e3))
