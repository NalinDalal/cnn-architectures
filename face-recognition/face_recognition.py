<VSCode.Cell language="python">
# Import Required Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL

# Set image data format
K.set_image_data_format('channels_last')

# Triplet Loss Function
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.maximum(tf.add(tf.subtract(pos_dist, neg_dist), alpha), 0)
    loss = tf.reduce_sum(basic_loss)
    return loss

# Load Pre-trained Model
from tensorflow.keras.models import model_from_json

def load_facenet_model():
    json_file = open('keras-facenet-h5/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('keras-facenet-h5/model.h5')
    return model

FRmodel = load_facenet_model()

# Function to Encode Images
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

# Face Verification
def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])
    if dist < 0.7:
        print(f"It's {identity}, welcome in!")
        return dist, True
    else:
        print(f"It's not {identity}, please go away")
        return dist, False

# Face Recognition
def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    identity = None

    for name, db_enc in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
        return min_dist, None
    else:
        print(f"It's {identity}, the distance is {min_dist}")
        return min_dist, identity

# Example Database
# database = {}
# database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
# database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
# database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
# database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
# database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
# database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
# database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
# database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
# database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
# database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
# database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
# database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)