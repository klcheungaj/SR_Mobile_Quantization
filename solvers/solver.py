import logging
from .networks import create_model
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import math
from utils import ProgressBar
import cv2
import os
import os.path as osp
import pickle

from tensorflow import keras
from tensorflow.keras.layers import Lambda, Input, InputLayer
from tensorflow.keras.models import Model, load_model

import tensorflow_model_optimization as tfmot

def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []
    def get_activations_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_anctivations):
        pass
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}

class Solver():
    def ps_quantization(self, layer):
        if 'lambda' in layer.name:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
        return layer

    def __init__(self, args, logger, train_data, val_data, val_data_raw, writer):
        super(Solver, self).__init__()
        self.opt = args['solver']
        self.qat = self.opt['qat']
        self.resume = self.opt['resume']
        self.lg = logger
        self.state = {'current_epoch': -1, 'best_epoch': -1, 'best_psnr': -1}
        
        if self.resume:
            self.lg.info('Load from checkpoint: [{}]'.format(self.opt['resume_path']))
            self.model = tf.keras.models.load_model(self.opt['resume_path'], custom_objects={'tf': tf, 'PSNR': PSNR})
            with open(args['paths']['state'], 'rb') as f:
                self.state = pickle.load(f)
                self.lg.info('Load checkpoint state successfully!')

        elif self.qat: #Quantization-Aware Training
            # load pretrain model
            self.lg.info('Loading pretrained model ...')
            p_model = tf.keras.models.load_model(self.opt['qat_path'], custom_objects={'tf': tf, 'PSNR': PSNR})
            self.lg.info('Start copying weights and annotate Lambda layer...')
            annotate_model = tf.keras.models.clone_model(
                p_model,
                clone_function=self.ps_quantization
            )
            self.lg.info('Start annotating other parts of model...')
            annotate_model = tfmot.quantization.keras.quantize_annotate_model(annotate_model)
            self.lg.info('Creating quantize-aware model...')
            depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, args['networks']['scale']))
            with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': depth_to_space, 'tf': tf}):
                self.model = tfmot.quantization.keras.quantize_apply(annotate_model)
        else:
            self.model = create_model(args['networks'])

        self.lg.info('Create model successfully! Params: [{:.2f}]K'.format(self.model.count_params()/1e3))

        self.train_data = train_data
        self.val_data = val_data
        self.val_data_raw = val_data_raw
        self.writer = writer
    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.opt['lr'])
        lr_scheduler = LearningRateScheduler(self.scheduler)
        epoch_end_call = Epoch_End_Callback(self.val_data, self.val_data_raw, self.train_data, self.lg, self.writer, args['paths'], self.opt['val_step'], state=self.state)
        self.callback = [lr_scheduler, epoch_end_call]        

    def train(self):
        if self.resume == False:
            self.model.compile(optimizer=self.optimizer, loss=self.opt['loss'], metrics=[PSNR])
        # history = self.model.fit(self.train_data, epochs=self.opt['epochs'], workers=self.opt['workers'], callbacks=self.callback, initial_epoch=self.state['current_epoch']+1, use_multiprocessing=True)
        history = self.model.fit(self.train_data, epochs=self.opt['epochs'], initial_epoch=self.state['current_epoch']+1, validation_data=self.val_data, use_multiprocessing=True, workers=self.opt['workers'], callbacks=self.callback)

    def scheduler(self, epoch):        
        if epoch in self.opt['lr_steps']:
            current_lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, current_lr*self.opt['lr_gamma'])
        return K.get_value(self.model.optimizer.lr)


class Epoch_End_Callback(Callback):
    def __init__(self, val_data, val_data_raw, train_data, lg, writer, paths, val_step, state):
        super(Epoch_End_Callback, self).__init__()
        self.val_step = val_step
        self.val_data = val_data
        self.val_data_raw = val_data_raw
        self.train_data = train_data
        self.lg = lg
        self.writer = writer
        self.ckp_path = paths['ckp']
        self.state_path = paths['state']
        self.root = paths['root']
        self.visual_path = paths['visual']
        self.best_epoch = state['best_epoch']
        self.best_psnr = state['best_psnr']

    def on_epoch_end(self, epoch, logs):
        # self.train_data.shuffle()
        if epoch % self.val_step != 0:
            return

        # validate
        psnr = 0.0
        pbar = ProgressBar(len(self.val_data_raw))
        for i, (lr, hr) in enumerate(self.val_data_raw):
            sr = self.model(lr)
            sr_numpy = K.eval(sr)
            hr_numpy = hr.numpy().astype(np.float32)
            # psnr += self.calc_psnr((sr_numpy).squeeze(), (hr).squeeze())
            psnr += PSNR(sr, hr_numpy).numpy()
            pbar.update('')
        print('\nvalidation completed')
        psnr = round(psnr / len(self.val_data_raw), 4)
        loss = round(logs['loss'], 4)

        # save best status
        if psnr >= self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = epoch
            self.model.save(self.ckp_path, overwrite=True, include_optimizer=True, save_format='tf')
            state = {
                'current_epoch': epoch,
                'best_epoch': self.best_epoch,
                'best_psnr': self.best_psnr
            }

            with open(self.state_path, 'wb') as f:
                pickle.dump(state, f)
            
        self.lg.info('Epoch: {:4} | PSNR: {:.2f} | Loss: {:.4f} | lr: {:.2e} | Best_PSNR: {:.2f} in Epoch [{}]'.format(epoch, psnr, loss, K.get_value(self.model.optimizer.lr), self.best_psnr, self.best_epoch))

        # record tensorboard
        self.writer.add_scalar('train_loss', loss, epoch)
        self.writer.add_scalar('val_psnr', psnr, epoch)


    def calc_psnr(self, y, y_target):
        h, w, c = y.shape
        y = np.clip(np.round(y), 0, 255).astype(np.float32)
        y_target = np.clip(np.round(y_target), 0, 255).astype(np.float32)

        # crop 1
        y_cropped = y[1:h-1, 1:w-1, :]
        y_target_cropped = y_target[1:h-1, 1:w-1, :]
        
        mse = np.mean((y_cropped - y_target_cropped) ** 2)
        if mse == 0:
            return 100
        return 20. * math.log10(255. / math.sqrt(mse))
