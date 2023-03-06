import os
import argparse
import cv2
import numpy as np
from options import parse
from solvers import Solver
from data import DIV2K
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
import os
import os.path as osp
from tensorboardX import SummaryWriter
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def random_crop(lowres_img, highres_img, patch_size=96, scale=4):
    """Crop images.

    low resolution images: 24x24
    high resolution images: 96x96
    """
    hr_crop_size = patch_size * scale
    lr_crop_size = patch_size
    lowres_img_shape = tf.shape(lowres_img)[:2]  # (height,width)

    lowres_width = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[1] - lr_crop_size + 1, dtype=tf.int32
    )
    lowres_height = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[0] - lr_crop_size + 1, dtype=tf.int32
    )

    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_img_cropped = lowres_img[
        lowres_height : lowres_height + lr_crop_size,
        lowres_width : lowres_width + lr_crop_size,
    ]  # 24x24
    highres_img_cropped = highres_img[
        highres_height : highres_height + hr_crop_size,
        highres_width : highres_width + hr_crop_size,
    ]  # 96x96

    return lowres_img_cropped, highres_img_cropped

def expand_dim(lowres_img, highres_img):    
    lowres_img_expand = tf.expand_dims(lowres_img, 0)
    highres_img_expand = tf.expand_dims(highres_img, 0)
    
    return lowres_img_expand, highres_img_expand
    # return tf.cast(lowres_img_expand, tf.float32), tf.cast(highres_img_expand, tf.float32)


def dataset_object(opt, dataset_cache, training=True, crop_batch=True):
    TRAIN_SIZE = 800
    REPEAT_SIZE = 25
    ds = dataset_cache
    if training:
        # Repeating Data, so that cardinality if dataset becomes infinte
        ds = ds.repeat(REPEAT_SIZE)
    
    if crop_batch:
        ds = ds.map(
            lambda lowres, highres: random_crop(lowres, highres, patch_size=opt['patch_size'], scale=opt['scale']),
            num_parallel_calls=AUTOTUNE,
        )
        # Batching Data
        ds = ds.batch(opt['batch_size'])
        
    else:
        ds = ds.map(
            lambda lowres, highres: expand_dim(lowres, highres),
            num_parallel_calls=AUTOTUNE,
        )
        
    print(f"data set length: {len(ds)}")

    # if training:
    #     ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
    #     ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)

    # prefetching allows later images to be prepared while the current image is being processed
    ds = ds.shuffle(int(TRAIN_SIZE*REPEAT_SIZE/opt['batch_size']))
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSRCNN Demo')
    parser.add_argument('--opt', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scale', default=None, type=int)
    parser.add_argument('--ps', default=48, type=int, help='patch_size')
    parser.add_argument('--bs', default=16, type=int, help='batch_size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--gpu_ids', default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default=None)
    parser.add_argument('--qat', action='store_true', default=False)
    parser.add_argument('--qat_path', default=None)

    args = parser.parse_args()
    args, lg = parse(args)

    # Tensorboard save directory
    resume = args['solver']['resume']
    tensorboard_path = 'Tensorboard/{}'.format(args['name'])

    if resume==False:
        if osp.exists(tensorboard_path):
            shutil.rmtree(tensorboard_path, True)
            lg.info('Remove dir: [{}]'.format(tensorboard_path))
    writer = SummaryWriter(tensorboard_path)

    # train_data = DIV2K(args['datasets']['train'])
    # lg.info('Create train dataset successfully!')
    # lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))
        
    # val_data = DIV2K(args['datasets']['val'])
    # lg.info('Create val dataset successfully!')
    # lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))
    div2k_data = tfds.image.Div2k(config="bicubic_x2")
    div2k_data.download_and_prepare()

    # Taking train data from div2k_data object
    train = div2k_data.as_dataset(split="train", as_supervised=True)
    train_cache = train.cache()
    # Validation data
    val = div2k_data.as_dataset(split="validation", as_supervised=True)
    val_cache = val.cache()
    train_data = dataset_object(args['datasets']['train'], train_cache, training=True, crop_batch=True)
    val_data = dataset_object(args['datasets']['val'], val_cache, training=False, crop_batch=True)
    val_data_raw = dataset_object(args['datasets']['val'], val_cache, training=False, crop_batch=False)

    # create solver
    lg.info('Preparing for experiment: [{}]'.format(args['name']))
    solver = Solver(args, lg, train_data, val_data, val_data_raw, writer)

    # train
    lg.info('Start training...')
    solver.train()
