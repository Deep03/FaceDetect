# Import dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow  
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import uuid


# directories:
POS_PATH = os.path.join('test', 'positive')
NEG_PATH = os.path.join('test', 'negative')
ANC_PATH = os.path.join('test', 'anchor')
