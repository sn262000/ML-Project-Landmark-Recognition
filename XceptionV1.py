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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.delf import image_input_fn,  get_features, match_images, get_delf_features_inliners_coordinates



landmark_classes = os.listdir("./data/train")
img_width, img_height = 224, 224
train_data_dir = './data/train'
validation_data_dir = './data/validation'
batch_size = 32
batch_size_small = 16
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


xception_base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, pooling='avg')
xception_model = Sequential()
xception_model.add(xception_base_model)
xception_model.add(Dense(512, activation='relu'))
#dropout layer, dropout randomly switches off some neurons in the network which forces the data to find new paths
xception_model.add(Dropout(0.5)) 
xception_model.add(Dense(n_classes, activation='softmax')) #class prediction(0–24)

# Say not to train first layer (Xception) model. It is already trained
xception_model.layers[0].trainable = False

# Pixel values rescaling from [0, 255] to [0, 1] interval
datagen_xception = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input)

# Retrieve images and their classes for train and validation sets
train_generator_xception = datagen_xception.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_small)

validation_generator_xception = datagen_xception.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_small)


# Compile the model 
xception_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0005),
              metrics=['accuracy'])


xception_model_checkpointer = ModelCheckpoint(filepath='./models/xception_model_checkpoint.h5', monitor='val_acc', verbose=1, save_best_only=True)

# Early stopping
# early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=5)

xception_model_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

hists = []

hist = xception_model.fit_generator(
    train_generator_xception,
    steps_per_epoch=train_samples // batch_size_small,
    epochs=1,
    verbose=1,
    callbacks=[xception_model_reducer, xception_model_checkpointer],
    validation_data=validation_generator_xception,
    validation_steps=validation_samples // batch_size_small)

hists.append(hist)
hist = xception_model.fit_generator(
    train_generator_xception,
    steps_per_epoch=train_samples // batch_size_small,
    epochs=10,
    verbose=1,
    callbacks=[xception_model_reducer, xception_model_checkpointer],
    validation_data=validation_generator_xception,
    validation_steps=validation_samples // batch_size_small)

hists.append(hist)


hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)
hist_df.index = np.arange(1, len(hist_df)+1)
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
axs[0].plot(hist_df.val_acc, lw=5, label='Validation')
axs[0].plot(hist_df.acc, lw=5, label='Training')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].grid()
axs[0].legend(loc=0)
axs[1].plot(hist_df.val_loss, lw=5, label='Validation')
axs[1].plot(hist_df.loss, lw=5, label='Training')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].grid()
axs[1].legend(loc=0)
plt.show();

# Compile the model 
xception_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])


hist = xception_model.fit_generator(
    train_generator_xception,
    steps_per_epoch=train_samples // batch_size_small,
    epochs=3,
    verbose=1,
    callbacks=[xception_model_reducer, xception_model_checkpointer],
    validation_data=validation_generator_xception,
    validation_steps=validation_samples // batch_size_small)

hists.append(hist)
