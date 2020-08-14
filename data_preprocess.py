import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
from PIL import Image
import matplotlib.patches as patches
import random

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications
from tensorflow.keras.applications import Xception, VGG16
from tensorflow.keras.applications.xception import preprocess_input

warnings.filterwarnings("ignore")
from utils.delf import image_input_fn,  get_features, match_images, get_delf_features_inliners_coordinates
landmark_classes = os.listdir("./data/train")
def removeInvalidImages(path):
    docList = os.listdir(path)
    for doc in docList:
        docPath = os.path.join(path,doc)
        if os.path.isfile(docPath):
            if os.path.getsize(docPath)<=(15*1024):
                os.remove(docPath)
        if os.path.isdir(docPath):
            listDoc(docPath)

img_width, img_height = 224, 224
train_data_dir = './data/train'
validation_data_dir = './data/validation'
batch_size = 32
batch_size_small = 16

removeInvalidImages(train_data_dir)
removeInvalidImages(validation_data_dir)

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

n_classes = 25 
train_samples = train_generator.samples
validation_samples = validation_generator.samples



