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

datagen_bottleneck = ImageDataGenerator(rescale=1./255)

# Retrieve images and their classes for train and validation sets
train_generator_bottleneck = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_small,
        class_mode=None,
        shuffle=False)

validation_generator_bottleneck = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_small,
        class_mode=None,
        shuffle=False)




model_vgg16 = VGG16(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet', pooling="max")

bottleneck_features_train = model_vgg16.predict_generator(train_generator_bottleneck, train_samples // batch_size, verbose=1)
np.save(open('./models/vgg16_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = model_vgg16.predict_generator(validation_generator_bottleneck, validation_samples // batch_size, verbose=1)
np.save(open('./models/vgg16_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

train_data = np.load(open('./models/vgg16_bottleneck_features_train.npy', 'rb'))

validation_data = np.load(open('./models/vgg16_bottleneck_features_validation.npy', 'rb'))

train_labels = to_categorical(train_generator_bottleneck.classes.tolist()[:train_data.shape[:1][0]], num_classes=n_classes)
validation_labels = to_categorical(validation_generator_bottleneck.classes.tolist()[:validation_data.shape[:1][0]], num_classes=n_classes)

model_top = Sequential()
model_top.add(Flatten(input_shape=train_data.shape[1:]))
model_top.add(Dense(512, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(n_classes, activation='softmax'))



model_top.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])


# Model saving callback
checkpointer_vgg16_bottleneck_features = ModelCheckpoint(filepath='./models/vgg16_bottleneck_features.h5', monitor='val_acc', verbose=1, save_best_only=True)

hists_vgg16_bottleneck_features = []

hist_vgg16_bottleneck_features = model_top.fit(
        train_data,
        train_labels,
        verbose=1,
        epochs=3, 
        batch_size=batch_size_small,
        callbacks=[checkpointer_vgg16_bottleneck_features],
        validation_data=(validation_data, validation_labels))

hists_vgg16_bottleneck_features.append(hist_vgg16_bottleneck_features)

early_stopping_vgg16_bottleneck_features = EarlyStopping(monitor='val_acc', verbose=1, patience=5)

hist_vgg16_bottleneck_features =  model_top.fit(
        train_data,
        train_labels,
        verbose=2,
        epochs=30, 
        batch_size=batch_size_small,
        callbacks=[checkpointer_vgg16_bottleneck_features, early_stopping_vgg16_bottleneck_features],
        validation_data=(validation_data, validation_labels))

hists_vgg16_bottleneck_features.append(hist_vgg16_bottleneck_features)

hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists_vgg16_bottleneck_features], sort=True)
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


base_model_vgg16 = VGG16(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet', pooling="max")

top_model = Sequential()

top_model.add(Flatten(input_shape=base_model_vgg16.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(n_classes, activation='softmax'))

top_model.load_weights('./models/vgg16_bottleneck_features.h5')


vgg16_model_tuning = Model(inputs=base_model_vgg16.input, outputs=top_model(base_model_vgg16.output))


vgg16_model_tuning.layers

for layer in vgg16_model_tuning.layers[:15]:
    layer.trainable = False

datagen_vgg16 = ImageDataGenerator(rescale=1./255)


train_generator_vgg16 = datagen_vgg16.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size_small)

validation_generator_vgg16 = datagen_vgg16.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size_small)


vgg16_model_tuning.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])


# Model saving callback
checkpointer_vgg16_tuning = ModelCheckpoint(filepath='./models/vgg16_tining.h5', monitor='val_acc', verbose=1, save_best_only=True)

reducer_vgg16_tuning = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

hists_vgg16_tuning  = []

hist_vgg16_tuning  = vgg16_model_tuning.fit_generator(
    train_generator_vgg16,
    steps_per_epoch=train_samples // batch_size_small,
    verbose=1,
    epochs=1, 
    callbacks=[checkpointer_vgg16_tuning, reducer_vgg16_tuning],
    validation_data=validation_generator_vgg16,
    validation_steps=validation_samples // batch_size_small
)

hists_vgg16_tuning.append(hist_vgg16_tuning)


hist_vgg16_tuning  = vgg16_model_tuning.fit_generator(
    train_generator_vgg16,
    steps_per_epoch=train_samples // batch_size_small,
    verbose=1,
    epochs=5, 
    callbacks=[checkpointer_vgg16_tuning, reducer_vgg16_tuning],
    validation_data=validation_generator_vgg16,
    validation_steps=validation_samples // batch_size_small
)

hists_vgg16_tuning.append(hist_vgg16_tuning)
hist_vgg16_tuning  = vgg16_model_tuning.fit_generator(
    train_generator_vgg16,
    steps_per_epoch=train_samples // batch_size_small,
    verbose=1,
    epochs=3, 
    callbacks=[checkpointer_vgg16_tuning, reducer_vgg16_tuning],
    validation_data=validation_generator_vgg16,
    validation_steps=validation_samples // batch_size_small
)

hists_vgg16_tuning.append(hist_vgg16_tuning)


hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists_vgg16_tuning], sort=True)
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


vgg16_model_tuning.load_weights("./models/vgg16_tining.h5")
vgg16_model_tuning.evaluate_generator(validation_generator_vgg16, validation_samples / batch_size_small, verbose=1)

def decode_predictions(original_array, model):
    predictions = model.predict(original_array)
    return landmark_classes[predictions.argmax()], predictions.max()

def load_image(image_path):
    original_image = Image.open(image_path).resize((img_width, img_height))
    original_array = np.expand_dims(np.array(original_image), 0)
    return original_array

def get_correct_predicted_image_for_class(class_name, model):
    dir_path = os.path.join(train_data_dir, class_name)
    filename = random.choice(os.listdir(dir_path))
    image = os.path.join(dir_path, filename)
    original_array = load_image(image)
    category, proba = decode_predictions(original_array, vgg16_model_tuning)
    if(class_name == category):
        return image
    else:
        return get_correct_predicted_image_for_class(class_name, model)



test_images_filenames = [
    './data/validation/arc_de_triomphe/2195175040.jpg',
    './data/validation/arc_de_triomphe/6045281055.jpg',
    './data/validation/arc_de_triomphe/3766337011.jpg',
    './data/validation/arc_de_triomphe/3769030741.jpg',
    './data/validation/big_ben/3947650648.jpg',
    './data/validation/big_ben/4860593771.jpg',
    './data/validation/big_ben/194760399.jpg',
    './data/validation/big_ben/490063476.jpg'
]

test_images_delf_features = get_features(test_images_filenames)


train_images_filenames = []
for line, filename in enumerate(test_images_filenames):    
    original_array = load_image(filename)
    category, proba = decode_predictions(original_array, vgg16_model_tuning)
    train_image_filename = get_correct_predicted_image_for_class(category, vgg16_model_tuning)
    train_images_filenames.append(train_image_filename)
    

    
train_images_delf_features = get_features(train_images_filenames)


n_lines = len(test_images_filenames)
plt.figure(figsize = (15, 4 * n_lines))

for line, filename in enumerate(test_images_filenames):    
    original_array = Image.open(filename)
    category, proba = decode_predictions(load_image(filename), vgg16_model_tuning)
    image_title = "'%s' %.1f%% confidence" % (category, proba * 100)
    
    train_image_filename = train_images_filenames[line]
    results_dict = {}
    results_dict[filename] = test_images_delf_features[filename]
    results_dict[train_image_filename] = train_images_delf_features[train_image_filename]
    min_x, max_x, min_y, max_y = get_delf_features_inliners_coordinates(results_dict, filename, train_image_filename)
   
    ax1 = plt.subplot(n_lines, 3, 3 * line + 1)

    # Display the image
    ax1.imshow(original_array)

    # Create a Rectangle patch
    rect = patches.Rectangle((min_x,min_y),max_x-min_x,max_y-min_y,linewidth=1,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax1.add_patch(rect)
    plt.title(image_title)
    
    image_prediction_array = Image.open(train_image_filename)
    ax2 = plt.subplot(n_lines, 3, 3 * line + 2)
    ax2.imshow(image_prediction_array)
    plt.title("Train Image for %s" % category)
    
    ax = plt.subplot(n_lines, 3, 3 * line + 3)
    match_images(results_dict, filename, train_image_filename, ax)
    

plt.tight_layout(pad = 4)




