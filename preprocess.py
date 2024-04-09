# Import dependencies
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow  
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import triplet_loss as tl
import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import facenet_model

DATA_DIR = os.path.dirname('test/')


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
    
    result = data_path + f'/{index}/{index}.png'
    if (result == 'None'):
        select_negdir(data_path, anchor_no)
    else:
        return result

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
    for folder_name in folder_data:
        arr = []
        folder = os.path.basename(folder_name)
        folder_dir = os.path.join(data_path, folder)
        anchor_path = select_dir(folder_dir)
        positive_path = select_dir(folder_dir)
        negative_path = select_negdir(data_path, int(folder))
        
        # Load and preprocess images
        arr.append(anchor_path)
        arr.append(positive_path)
        arr.append(negative_path)
        batch.append(arr)
    return batch

triplet_batch = triplet_selection(DATA_DIR)


# datagen = ImageDataGenerator(rescale=1./255)

# Assuming your triplets are in a list called 'triplets'
def load_images(triplets):
    images = []
    for triplet in triplets:
        img1 = img_to_array(load_img(triplet[0], target_size=(112, 112)))
        img2 = img_to_array(load_img(triplet[1], target_size=(112, 112)))
        img3 = img_to_array(load_img(triplet[2], target_size=(112, 112)))
        images.append([img1, img2, img3])  # Append instead of extend to keep triplets intact
    return np.array(images)

images = load_images(triplet_batch)
x_train = images  



def compute_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def compute_embeddings(images, model):
    # Normalize pixel values to [0, 1]
    images = images.astype('float32') / 255.0
    # Obtain embeddings by passing images through the model
    embeddings = model.predict(images)
    return embeddings

def prepare_labels(triplets, model):
    labels = []
    for triplet in triplets:
        # Extract anchor, positive, and negative images from the triplet
        anchor_img = triplet[0]
        positive_img = triplet[1]
        negative_img = triplet[2]
        # Reshape images to match model's input shape if necessary
        # (Assuming the model expects input shape of (112, 112, 3))
        anchor_img = np.expand_dims(anchor_img, axis=0)
        positive_img = np.expand_dims(positive_img, axis=0)
        negative_img = np.expand_dims(negative_img, axis=0)
        # Compute embeddings for anchor images
        anchor_embedding = compute_embeddings(anchor_img, model)
        # You can similarly compute embeddings for positive and negative images if needed
        positive_embedding = compute_embeddings(positive_img, model)
        negative_embedding = compute_embeddings(negative_img, model)
        # For simplicity, let's assume positive_embedding and negative_embedding are available
        # Compute distances between anchor-positive and anchor-negative
        distance_positive = compute_distance(anchor_embedding, positive_embedding)
        distance_negative = compute_distance(anchor_embedding, negative_embedding)
        label = 1 if distance_positive < distance_negative else 0
        labels.append(anchor_embedding)  # Append anchor embeddings
    return np.array(labels)

# Assuming you have your FaceNet model stored in `model`
# And your triplets organized in `triplet_batch`
y_train = prepare_labels(x_train, facenet_model)