import base64
from io import BytesIO
import math
import numpy as np
import tensorflow as tf
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 自己实现的卷积函数
def conv2d(input, filter, stride, padding):
    # batch x height x width x channels
    in_s = input.shape
    # height x width x in_channels x out_channels
    f_s = filter.shape

    temp = []

    assert len(in_s) == 4, 'input size rank 4 required!'
    assert len(f_s) == 4, 'filter size rank 4 required!'
    assert f_s[2] == in_s[3], 'intput channels not match filter channels.'
    assert f_s[0] >= stride and f_s[1] >= stride, 'filter should not be less than stride!'
    assert padding in ['SAME', 'VALID'], 'padding value[{0}] not allowded!!'.format(padding)

    if padding != 'VALID':
        # tf官网的定义为padding=same的时候out_shape = math.ceil(in_shape / stride)
        # padding=valid的时候out_shape = math.ceil((in_shape - f_shape + 1 / stride))
        temp = np.array(in_s[1: 3]) / stride
    else:
        temp = (np.array(in_s[1: 3]) - np.array(f_s[: 2]) + 1) / stride
    out_shape = (math.ceil(temp[0]), math.ceil(temp[1]))
    out_shape = np.concatenate([in_s[:1], out_shape, f_s[-1:]])
    output = np.zeros(out_shape, dtype=np.int64)
    # 计算padding
    # out = (in - f + 2p) / stride + 1
    # 2p = (out - 1) * stride - in + f
    _2p = np.array(out_shape[1: 3] - 1) * stride - \
        np.array(in_s[1: 3]) + np.array(f_s[: 2])
    # 这里tensorflow的卷积居然是上左padding分配了1 右下分配了2 一开始写成 上左2 下右边1 纳闷了半天
    lp = np.array(_2p) // 2
    rp = np.array(_2p) - np.array(lp)
    input2 = input
    if(lp.all()>0 or rp.all()>0):
        input2 = np.pad(input, ((0, 0), (lp[0], rp[0]), (lp[1], rp[1]), (0, 0)), 'constant')        
    in_s = input2.shape
    print("padding shape:", in_s, "; left padding: ", lp, "; right padding: ", rp)
    # 循环每个卷积核
    for kernel in range(f_s[3]):
        out_r = 0
        # 逐行扫描，每次行数叠加stride，直到越界
        for row in range(0, in_s[1], stride):
            if(row+f_s[0] - 1 >= in_s[1]):
                break
            # 新的行迭代、列回到0
            out_c = 0
            # 每行逐列扫描，每次列数叠加stride，直到越界
            for col in range(0, in_s[2], stride):
                if(col+f_s[1] - 1 >= in_s[2]):
                    break
                # print([row+f_s[0], col+f_s[1]])
                # 提取原图的卷积核覆盖范围
                cover = input2[:, row:row+f_s[0], col:col+f_s[1], :]
                output[:, out_r, out_c, kernel] = np.sum(cover * filter[:, :, :, kernel])
                out_c += 1
            # 每次行迭代，feature map对应行加1
            out_r += 1
    return output

def conv2d_depthwise(input, filter, stride, padding):
    # batch x height x width x channels
    in_s = input.shape
    # height x width x channel
    f_s = filter.shape

    temp = []

    assert len(in_s) == 4, 'input size rank 4 required!'
    assert len(f_s) == 3, 'filter size rank 3 required!'
    assert f_s[2] == in_s[3], 'intput channels not match filter channels.'
    assert f_s[0] >= stride and f_s[1] >= stride, 'filter should not be less than stride!'
    assert padding in ['SAME', 'VALID'], 'padding value[{0}] not allowded!!'.format(padding)

    if padding != 'VALID':
        # tf官网的定义为padding=same的时候out_shape = math.ceil(in_shape / stride)
        # padding=valid的时候out_shape = math.ceil((in_shape - f_shape + 1 / stride))
        temp = np.array(in_s[1: 3]) / stride
    else:
        temp = (np.array(in_s[1: 3]) - np.array(f_s[: 2]) + 1) / stride
    out_shape = (math.ceil(temp[0]), math.ceil(temp[1]))
    out_shape = np.concatenate([in_s[:1], out_shape, f_s[-1:]])
    output = np.zeros(out_shape, dtype=np.int64)
    # 计算padding
    # out = (in - f + 2p) / stride + 1
    # 2p = (out - 1) * stride - in + f
    _2p = np.array(out_shape[1: 3] - 1) * stride - \
        np.array(in_s[1: 3]) + np.array(f_s[: 2])
        
    lp = np.array(_2p) // 2
    rp = np.array(_2p) - np.array(lp)
    input2 = input
    if(lp.all()>0 or rp.all()>0):
        input2 = np.pad(input, ((0, 0), (lp[0], rp[0]), (lp[1], rp[1]), (0, 0)), 'constant')        
    in_s = input2.shape
    print("padding shape:", in_s, "; left padding: ", lp, "; right padding: ", rp)
    # 循环每个卷积核
    for kernel in range(f_s[2]):
        out_r = 0
        # 逐行扫描，每次行数叠加stride，直到越界
        for row in range(0, in_s[1], stride):
            if(row+f_s[0] - 1 >= in_s[1]):
                break
            # 新的行迭代、列回到0
            out_c = 0
            # 每行逐列扫描，每次列数叠加stride，直到越界
            for col in range(0, in_s[2], stride):
                if(col+f_s[1] - 1 >= in_s[2]):
                    break
                # print([row+f_s[0], col+f_s[1]])
                # 提取原图的卷积核覆盖范围
                cover = input2[:, row:row+f_s[0], col:col+f_s[1], kernel]
                output[:, out_r, out_c, kernel] = np.sum(cover * filter[:, :, kernel])
                out_c += 1
            # 每次行迭代，feature map对应行加1
            out_r += 1
    return output

