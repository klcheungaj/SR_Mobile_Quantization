import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add, DepthwiseConv2D, SeparableConv2D, MaxPooling2D
from tensorflow.keras.activations import gelu, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K
import numpy as np
from keras_flops import get_flops
import math

def RegConv(inputs, in_ch, out_ch, kernel_size=3, stride=1,
                dilation=1, bias=True, padding="same", p=0.25, min_mid_channels=4):
    # pointwise
    x = Conv2D(
            filters=out_ch,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            groups=1,
            use_bias=bias,
    )(inputs)
    return x

def DepthWiseConv(inputs, in_ch, out_ch, kernel_size=3, stride=1,
                dilation=1, bias=True, padding="same", p=0.25, min_mid_channels=4):
    # Depthwise separable 2D convolution
    out = SeparableConv2D(
            filters=out_ch,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            use_bias=bias
    )(inputs)
    return out

def BSConvU(inputs, in_ch, out_ch, kernel_size=3, stride=1,
                dilation=1, bias=True, padding="same", p=0.25, min_mid_channels=4):
    # pointwise
    x = Conv2D(
            filters=out_ch,
            kernel_size=1,
            strides=1,
            padding='valid',
            dilation_rate=1,
            groups=1,
            use_bias=False,
    )(inputs)

    # depthwise
    x = DepthwiseConv2D (
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            use_bias=bias
    )(x)
    return x

def BSConvS(inputs, in_ch, out_ch, kernel_size=3, stride=1,
                dilation=1, bias=True, padding="same", p=0.25, min_mid_channels=4):
    assert 0.0 <= p <= 1.0
    mid_channels = min(in_ch, max(min_mid_channels, math.ceil(p * in_ch)))
    # pointwise 1
    x = Conv2D(
            filters=mid_channels,
            kernel_size=1,
            strides=1,
            padding='valid',
            dilation_rate=1,
            groups=1,
            use_bias=False,
    )(inputs)

    # pointwise 2
    x = Conv2D(
            filters=out_ch,
            kernel_size=1,
            strides=1,
            padding='valid',
            dilation_rate=1,
            groups=1,
            use_bias=False,
    )(x)

    # depthwise
    x = DepthwiseConv2D (
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            use_bias=bias
    )(x)
    return x

def ESA(inputs, num_feat, conv=RegConv, p=0.25):
    f = num_feat // 4
    c1_ = Conv2D(filters=f, kernel_size=1, padding='same')(inputs)
    c1 = conv(in_ch=f, out_ch=f, kernel_size=3, stride=2, padding='same')(c1_)
    v_max = MaxPooling2D(pool_size=7, strides=3, padding='same')(c1)
    v_range = gelu(conv(in_ch=f, out_ch=f, kernel_size=3, stride=3, padding='same')(v_max))
    c3 = gelu(conv(in_ch=f, out_ch=f, kernel_size=3, paddng='same')(v_range))
    c3 = conv(in_ch=f, out_ch=f, kernel_size=3, padding='same')(c3)
    cf = Conv2D(filters=f, kernel_size=1, padding='same')(c1_)
    c4 = Conv2D(filter=num_feat, kernel_size=1)(c3 + cf)
    m = sigmoid(c4)
    return input * m

def CCALayer(inputs, channel, reduction=16):
    y = 

def ESDB(inputs, in_ch, out_ch, conv=RegConv, p=0.25):
    dc = distilled_channels = in_ch // 2
    rc = remaining_channels = in_ch

    act = gelu
    distilled_c1 = act(Conv2D(filters=dc, kernel_size=1)(inputs))
    r_c1 = conv(inputs=inputs, in_ch=in_ch, out_ch=rc, kernel_size=3)
    r_c1 = act(r_c1 + inputs)

    distilled_c2 = act(Conv2D(filters=dc, kernel_size=1)(r_c1))
    r_c2 = conv(inputs=r_c1, in_ch=remaining_channels, out_ch=rc, kernel_size=3)
    r_c2 = act(r_c2 + r_c1)

    distilled_c3 = act(Conv2D(filters=dc, kernel_size=1)(r_c2))
    r_c3 = conv(inputs=r_c2, in_ch=remaining_channels, out_ch=rc, kernel_size=3)
    r_c3 = act(r_c3 + r_c2)

    r_c4 = act(conv(inputs=r_c3, in_ch=remaining_channels, out_ch=dc, kernel_size=3))
    
    out = tf.concat([distilled_c1, distilled_c2, distilled_c3, r_c4], 1)
    out = Conv2D(filters=in_ch, kernel_size=1)(out)
    out_fused = ESA(inputs=out, num_feat=in_ch, conv=conv)
    out_fused = cca(out_fused)
    return out_fused + input


def BSRN(upscale=2, num_feat=64, num_block=8, in_ch=3, out_ch=3,
            conv='BSConvU', upsampler='pixelshuffledirect', p=0.25):
    inp = Input(shape=(None, None, 3)) 
    conv = Conv2D   ## TODO: add parameter
    if conv == 'DepthWiseConv':
        conv = DepthWiseConv
    elif conv == 'BSConvU':
        conv = BSConvU
    elif conv == 'BSConvS':
        conv = BSConvS
    else:
        conv = RegConv
    
    # Feature extraction
    x = conv(filters=num_feat, filter_size=3)(inp)
    x = ESDB(out_channels=num_feat, conv=conv, p=p)(x)
    x = ESDB(out_channels=num_feat, conv=conv, p=p)(x)
    x = ESDB(out_channels=num_feat, conv=conv, p=p)(x)
    x = ESDB(out_channels=num_feat, conv=conv, p=p)(x)
    x = ESDB(out_channels=num_feat, conv=conv, p=p)(x)
    x = ESDB(out_channels=num_feat, conv=conv, p=p)(x)
    x = ESDB(out_channels=num_feat, conv=conv, p=p)(x)
    x = Conv2D(filters=num_feat, filter_size=1, activation='gelu')(x)
    x = Conv2D(filters=num_feat, filter_size=3)(x)
    
    if upsampler == 'pixelshuffledirect':
        x = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=out_ch)
    else:
        raise NotImplementedError(("Check the Upsampeler. None or not support yet"))
    
    model = Model(inputs=inp, outputs=out)
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    return model

if __name__ == '__main__':
    model = base7()
    print('Params: [{:.2f}]K'.format(model.count_params()/1e3))

