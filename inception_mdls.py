from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_slim as slim
import tensorflow as tf

import keras

# Load InceptionV3 without the top classification layer
def inception_net(input_tensor):
    inception_layer = keras.applications.InceptionV3(
        include_top=False,
        weights='None',  # You can specify 'None' to start with random weights
        input_tensor=input_tensor, # (n - 1) tensor
        input_shape=(299, 299, 3),  # Adjust input shape based on your requirements,
    )

    return inception_layer
