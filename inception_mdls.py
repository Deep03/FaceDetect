from __future__ import absolute_import
import tf_slim as slim
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense, BatchNormalization


# FIX NEEDED: Add inception v3 layer to improve model
def inception_v3(input_shape):
    inception_layer = keras.applications.InceptionV3(
        include_top=False,
        weights=None,  
        input_shape=input_shape
    )

    return inception_layer


# derived from Going deeper with convolutions: https://arxiv.org/pdf/1409.4842.pdf
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name):
    conv1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', name=name+'_1x1')(x)

    conv3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', name=name+'_3x3_reduce')(x)
    conv3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', name=name+'_3x3')(conv3x3_reduce)

    conv5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', name=name+'_5x5_reduce')(x)
    conv5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', name=name+'_5x5')(conv5x5_reduce)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name=name+'_pool')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', name=name+'_pool_proj')(pool_proj)

    inception_module = concatenate([conv1x1, conv3x3, conv5x5, pool_proj], axis=-1, name=name+'_concat')

    return inception_module