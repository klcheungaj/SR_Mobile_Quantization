import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal, he_normal
import tensorflow.keras.backend as K
import numpy as np
import tensorflow_models as tfm


def base7(scale=2, in_channels=3, num_fea=28, m=4, out_channels=3):
    inp = Input(shape=(None, None, 3)) 
    upsample_func = Lambda(lambda x_list: tf.concat(x_list, axis=3))
    upsampled_inp = upsample_func([inp]*(scale**2))

    # Feature extraction
    x = SeparableConv2D(num_fea, 3, padding='same', activation='relu', depthwise_initializer=he_normal(), pointwise_initializer=he_normal(), bias_initializer='zeros')(inp)

    for i in range(m):
        x = SeparableConv2D(num_fea, 3, padding='same', activation='relu', depthwise_initializer=he_normal(), pointwise_initializer=he_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = SeparableConv2D(out_channels*(scale**2), 3, padding='same', activation='relu', depthwise_initializer=he_normal(), pointwise_initializer=he_normal(), bias_initializer='zeros')(x)
    x = SeparableConv2D(out_channels*(scale**2), 3, padding='same', activation='relu', depthwise_initializer=he_normal(), pointwise_initializer=he_normal(), bias_initializer='zeros')(x)
    x = Add()([upsampled_inp, x])
    
    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    out = depth_to_space(x)
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(out)
    
    model = Model(inputs=inp, outputs=out)
    print("try_count_flops report: {} GFLOPS".format(tfm.core.train_utils.try_count_flops(model)/1e9))
    return model

if __name__ == '__main__':
    model = base7()
    print('Params: [{:.2f}]K'.format(model.count_params()/1e3))
