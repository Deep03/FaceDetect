# Import dependencies
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow  
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import uuid
import triplet_loss as tl

DATA_DIR = os.path.dirname('test/')


# script to refactor the file labels
# def refactor_label(file_path):
#     folder_name = os.path.basename(file_path)
#     dir_name = file_path + '/'
#     arr = os.listdir(dir_name)
#     for file_name in arr:
#         new_file_name = folder_name + '_' + file_name
#         rename_dir = file_path + '/' + new_file_name
#         os.rename(dir_name + rename_dir, rename_dir)

# refactor_label('test/100')


# select directory for positve and negative pngs
def select_dir(data_path):
    """
    data_path: DATA_DIR + 'folder_name"
    EX: test/61

    Random anchor and positive
    """
    index = random.randint(0, 71)
    return data_path + f'/{index}.png'


def select_negdir(data_path, anchor_no):
    """
    data_path: DATA_DIR
    EX: test

    Recursive call to make sure negative png is not of the same person
    """
    index = random.randint(0, 71)
    if index == anchor_no:
        select_negdir(data_path, anchor_no)
    
    else:
        return data_path + f'/{index}/{index}.png'

def triplet_selection(data_path):
    """
    batch: batches of triplets(anchor, positive, negative), each item is a triplet
    arr: triplet, arr of length 3
    folder_data: array of all folders inside data_path 
    folder_dir: path including inside folder
    EX: test/25

    """
    batch = []
    folder_data = os.listdir(data_path)
    # folder_data.remove('.DS_Store')
    for folder_name in folder_data:
        arr = []
        folder = os.path.basename(folder_name)
        folder_dir = os.path.join(data_path, folder)
        anchor_path = select_dir(folder_dir)
        positive_path = select_dir(folder_dir)
        negative_path = select_negdir(data_path, int(folder))
        arr.append(anchor_path)
        arr.append(positive_path)
        arr.append(negative_path)
        batch.append(arr)
        


triplet_selection(DATA_DIR)