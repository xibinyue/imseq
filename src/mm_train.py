#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author:  xibin.yue   
date:  2016/12/7
descrption: 
"""
import os
import sys

sys.path.append(os.path.expanduser("~/libs"))

import random
import string
import numpy as np
import tensorflow as tf
import tensortools as tt

from model.frame_prediction import LSTMConv2DPredictionModel

INPUT_SEQ_LENGTH = 10
OUTPUT_SEQ_LENGTH = 10

# model
WEIGHT_DECAY = 1e-5
CONV_FILTERS = [32, 64, 64]
CONV_KSIZES = [(5, 5), (3, 3), (3, 3)]
CONV_STRIDES = [(2, 2), (1, 1), (2, 2)]
CONV_BN = True
CONV_BIAS = 0.1
OUTPUT_ACTIV = tf.nn.sigmoid
LSTM_LAYERS = 2
LSTM_KSIZE_INPUT = (3, 3)
LSTM_KSIZE_HIDDEN = (5, 5)
LSTM_PEEPHOLES = True
MAIN_LOSS = tt.loss.bce
MAIN_LOSS_ALPHA = 1.0
GDL_LOSS_ALPHA = 1.0
SSIM_LOSS_ALPHA = 0.0
SCHED_SAMPLING_DECAY = 1000.0

# optimizer
LR_INIT = 0.001
LR_DECAY_INTERVAL = 1000
LR_DECAY_FACTOR = 0.95

# training
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 50
MAX_STEPS = 100000

# validation while training
KEEP_CHECKPOINTS = 20
OUT_DIR_NAME = "out-train"
NUM_SAMPLES = 4
GIF_FPS = 5

ROOT_DIR = "/work/sauterme/"
DATA_DIR = ROOT_DIR + "data"
TRAIN_DIR_BASE = os.path.join(ROOT_DIR, "train", "mm",
                              "{}".format("ss" if SCHED_SAMPLING_DECAY is not None else "as"),
                              "{}l{}i{}h{}".format(LSTM_LAYERS, LSTM_KSIZE_INPUT[0], LSTM_KSIZE_HIDDEN[0],
                                                   "p" if LSTM_PEEPHOLES else ""),
                              "c{}k{}s{}{}".format("".join([str(f) for f in CONV_FILTERS]),
                                                   "".join([str(k[0]) for k in CONV_KSIZES]),
                                                   "".join([str(s[0]) for s in CONV_STRIDES]),
                                                   "bn" if CONV_BN else ""),
                              "wd{:.0e}".format(WEIGHT_DECAY))

# optional comment-word
comment = ""
if comment is not None or comment != "":
    TRAIN_DIR_BASE = os.path.join(TRAIN_DIR_BASE, comment)

TRAIN_DIR = os.path.join(TRAIN_DIR_BASE,
                         "".join(random.choice(string.ascii_uppercase) for _ in range(2)))

# check for conflict
print("Training directory  : {}".format(TRAIN_DIR))
print("Is new training     : {}".format(not os.path.exists(TRAIN_DIR_BASE)))

assert not os.path.exists(TRAIN_DIR)

AS_BINARY = True if MAIN_LOSS.__name__ == 'bce' else False
print("MovingMNIST as binary data: {}".format(AS_BINARY))

dataset_train = tt.datasets.moving_mnist.MovingMNISTTrainDataset(DATA_DIR,
                                                                 input_shape=[INPUT_SEQ_LENGTH, 64, 64, 1],
                                                                 target_shape=[OUTPUT_SEQ_LENGTH, 64, 64, 1],
                                                                 as_binary=AS_BINARY)
dataset_valid = tt.datasets.moving_mnist.MovingMNISTValidDataset(DATA_DIR,
                                                                 input_shape=[INPUT_SEQ_LENGTH, 64, 64, 1],
                                                                 target_shape=[OUTPUT_SEQ_LENGTH, 64, 64, 1],
                                                                 as_binary=AS_BINARY)

GPU_ID = 0
runtime = tt.core.DefaultRuntime(train_dir=TRAIN_DIR, gpu_devices=[GPU_ID])

runtime.register_datasets(dataset_train, dataset_valid)
runtime.register_model(LSTMConv2DPredictionModel(weight_decay=WEIGHT_DECAY,
                                                 filters=CONV_FILTERS, ksizes=CONV_KSIZES,
                                                 strides=CONV_STRIDES, bias_init=CONV_BIAS,
                                                 output_activation=OUTPUT_ACTIV,
                                                 bn_feature_enc=CONV_BN, bn_feature_dec=CONV_BN,
                                                 lstm_layers=LSTM_LAYERS,
                                                 lstm_ksize_input=LSTM_KSIZE_INPUT,
                                                 lstm_ksize_hidden=LSTM_KSIZE_HIDDEN,
                                                 lstm_use_peepholes=LSTM_PEEPHOLES,
                                                 scheduled_sampling_decay_rate=SCHED_SAMPLING_DECAY,
                                                 main_loss=MAIN_LOSS,
                                                 alpha_main_loss=MAIN_LOSS_ALPHA,
                                                 alpha_gdl_loss=GDL_LOSS_ALPHA,
                                                 alpha_ssim_loss=SSIM_LOSS_ALPHA))
runtime.register_optimizer(tt.training.Optimizer(tt.training.ADAM,
                                                 LR_INIT,
                                                 LR_DECAY_INTERVAL,
                                                 LR_DECAY_FACTOR))

runtime.build(max_checkpoints_to_keep=KEEP_CHECKPOINTS)


def write_animations(rt, dataset, gstep):
    root = os.path.join(rt.train_dir, OUT_DIR_NAME, "{:06d}".format(gstep))
    x, y = dataset.get_batch(NUM_SAMPLES)
    pred = rt.predict(x)

    for i in range(NUM_SAMPLES):
        concat_y = np.concatenate((x[i], y[i]))
        concat_pred = np.concatenate((x[i], pred[i]))

        tt.utils.video.write_multi_gif(os.path.join(root, "anim-{:02d}.gif".format(i)),
                                       [concat_y, concat_pred],
                                       fps=GIF_FPS, pad_value=1.0)

        tt.utils.video.write_multi_image_sequence(os.path.join(root, "timeline-{:02d}.png".format(i)),
                                                  [concat_y, concat_pred],
                                                  pad_value=1.0)


def on_valid(rt, gstep):
    write_animations(rt, rt.datasets.valid, gstep)


runtime.train(BATCH_SIZE, EVAL_BATCH_SIZE, steps=MAX_STEPS, on_validate=on_valid,
              checkpoint_steps=2000)

runtime.validate(EVAL_BATCH_SIZE)
runtime.close()
