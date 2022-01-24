import numpy as np
import os
import time
import glob
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import PIL
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt


def cornerDetection() :
    DATASET_PATH  = r"C:\Users\Windows\Desktop\Hw5\sample"
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 2
    BATCH_SIZE = 48
    FREEZE_LAYERS = 48
    NUM_EPOCHS = 5
    WEIGHTS_FINAL = 'model-resnet50-final.h5'
    train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
    train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

    valid_datagen = ImageDataGenerator()
    valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)
    for cls, idx in train_batches.class_indices.items():
        print('Class #{} = {}'.format(idx, cls))
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    net_final.compile(optimizer=adam_v2.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print(net_final.summary())

def intrinsicMatrix():
    path_photo=r"C:\Users\Windows\Desktop\Hw5\flow.png"
    img_2 = cv2.imread(path_photo)
    cv2.imshow('tensorboard',img_2)

def distortionMatrix():
    left = np.array(['Before Random-Erasing','After Random-Erasing'])
    height = np.array([74.5,78.9])
    plt.ylim([70,80])
    plt.ylabel("accuracy")
    plt.bar(left, height,width=0.5)
    plt.show()
def findextrinsic(num):
	
    new_model = tf.keras.models.load_model(r"C:\Users\Windows\Desktop\energy.h5")
    path_predict=str("C:/Users/Windows/Desktop/Hw5/sample/train/cats/"+str(int(num))+".JPG")
    img = cv2.imread(path_predict)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, 0)
    if new_model.predict(img)[0][0] > new_model.predict(img)[0][1]:
        img_2 = cv2.imread(path_predict)
        cv2.imshow('Class:cat',img_2)
    else :
        img_2 = cv2.imread(path_predict)
        cv2.imshow('Class:dog',img_2)
