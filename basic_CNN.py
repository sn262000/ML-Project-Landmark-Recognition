#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf



train_datagen= ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip= True)
valid_datagen= ImageDataGenerator(rescale=1./255)

train_set= train_datagen.flow_from_directory('./dataset/train', 
                                             target_size=(80,80), batch_size=28, class_mode='categorical', 
                                             shuffle=True, seed=20)
valid_set= valid_datagen.flow_from_directory('./dataset/validation', 
                                             target_size=(80,80), batch_size=22, class_mode='categorical', 
                                             shuffle=False)


tf.compat.v1.disable_eager_execution()
model=Sequential()
model.add(Conv2D(64, (3,3), input_shape=(80,80,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=25, activation='softmax'))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])
step_size_train=train_set.n//train_set.batch_size
step_size_valid=valid_set.n//valid_set.batch_size


model.fit_generator(
        train_set,
        steps_per_epoch=1382,
        epochs=5,
        verbose=1,
        validation_data=valid_set,
        validation_steps=382)
    


import os
img=image.load_img('as.jpg', target_size=(80,80))
plt.imshow(img)
plt.show()
img=image.img_to_array(img)
img=img.reshape(1,80,80,3)
pred=model.predict(img)
print('Landmark_1: ',pred[0][0])
print('Landmark_2: ',pred[0][1])
print('Landmark_3: ',pred[0][2])
print('Landmark_4: ',pred[0][3])
print('Landmark_5: ',pred[0][4])
print('Landmark_6: ',pred[0][5])
   




