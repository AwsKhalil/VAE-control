#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:17:08 2024

@author: aws
"""
# %% Importing Libraries and modules 
import numpy as np
from numpy.random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import pickle as pkl
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.resnet50 import ResNet50
import datetime
import yaml
# My Modules
# from drive_data import DriveData
from process_data_vae_new import ProcessData
from process_data_vae_new import check_data, collect_dataset, build_filtered_dataset
from image_process import ImageProcess
import gpu_options


# %%
# clear seesion if needed.
# tf.compat.v1.keras.backend.clear_session()

config_path = '/home/aws/latency_mitigaion_vae/2024/config/'
with open(config_path+'config.yaml') as file:
# with open('config.yaml') as file:
    global config
    config = yaml.load(file, Loader=yaml.FullLoader)

gpu_options.set()

# %% Load DATA and check

data_path = config['data_path']
processed_data = ProcessData(data_path, 
                             balance_data=config['steering_data_balancing'])
processed_data.build_datasets(augment=config['augment'])
if config['check_datasets']:
    check_data(processed_data)

# %% define your data
# These are the datasets that you create from all available data in the drive data CSV file.
# later on you filter and structure your data to be ready for training ( using build_filtered_dataset() )
train_data = processed_data.train_dataset
valid_data = processed_data.valid_dataset
test_data = processed_data.test_dataset

# %% Collect Datasets
# Collect the entire datasets if you want to create plots agianst time or use data separately. 
# Please note that setting config['collect_datasets'] to True significantly slows the process, so only use it if needed.
if config['collect_datasets']:
    img_train, vel_train, yaw_rate_train, head_train, steer_train, thr_brk_train = collect_dataset(train_data, 'training')
    img_valid, vel_valid, yaw_rate_valid, head_valid, steer_valid, thr_brk_valid = collect_dataset(valid_data, 'validation')
    img_test, vel_test, yaw_rate_test, head_test, steer_test, thr_brk_test = collect_dataset(test_data, 'testing')

    print("img_train.shape {0} \/ img_valid.shape {1} \/ img_test.shape {2}".format(
        img_train.shape, img_valid.shape, img_test.shape))

    print("vel_train.shape {0} \/ vel_valid.shape {1} \/ vel_test.shape {2}".format(
        vel_train.shape, vel_valid.shape, vel_test.shape))

    print("yaw_rate_train.shape {0} \/ yaw_rate_valid.shape {1} \/ yaw_rates_test.shape {2}".format(
        yaw_rate_train.shape, yaw_rate_valid.shape, yaw_rate_test.shape))

    print("head_train.shape {0} \/ head_valid.shape {1} \/ head_test.shape {2}".format(
        head_train.shape, head_valid.shape, head_test.shape))

    print("steer_train.shape {0} \/ steer_valid.shape {1} \/ steer_test.shape {2}".format(
        steer_train.shape, steer_valid.shape, steer_test.shape))

    print("thr_brk_train.shape {0} \/ thr_brk_valid.shape {1} \/ thr_brk_test.shape {2}".format(
        thr_brk_train.shape, thr_brk_valid.shape, thr_brk_test.shape))



# %% Create the filtered datasets
# These are the datasets that you will use for training
# they will be filtered based on your config file
# in the config file determin your inputs and outputs.

# in: img, vel  # out: img_reconstructed.

# Apply filtering, caching, shuffling, batching, and prefetching to the train dataset
train_data_filtered = build_filtered_dataset(train_data)
train_data_filtered = train_data_filtered.cache()  # Cache the data in memory if possible
train_data_filtered = train_data_filtered.shuffle(1000)  # Shuffle the data (buffer size can be adjusted)
train_data_filtered = train_data_filtered.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for faster loading

# Apply the same steps for the validation dataset (no need to shuffle)
valid_data_filtered = build_filtered_dataset(valid_data)
valid_data_filtered = valid_data_filtered.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for faster loading

# Apply the same steps for the testing dataset (no need to shuffle)
test_data_filtered = build_filtered_dataset(test_data)
test_data_filtered = test_data_filtered.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for faster loading



