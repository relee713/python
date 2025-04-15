# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import datetime
# cv is computer vision package
import cv2 as cv

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("trainset shape: ", train_images.shape)
print("testset shape: ", test_images.shape)

# Find the indices of images with label '8'
indices = [i for i, label in enumerate(train_labels) if label == 8]
print(indices)
print(train_labels[17])

num_images_to_print = 5
for i in range(num_images_to_print):
    index = indices[i]
    image = train_images[index]
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {train_labels[index]}")
    plt.axis('off')
    plt.show()
    
image = train_images[1]
#image = image.reshape(28, 28)
plt.imshow(image, cmap='gray')
plt.show()

print ("train_images.shape: ",train_images.shape) #Q2-3-0
print(train_labels[:10]) #Q2-3-0

print("test_images.shape", test_images.shape)
print("len(test_labels)", len(test_labels))
print("test_labels", test_labels)