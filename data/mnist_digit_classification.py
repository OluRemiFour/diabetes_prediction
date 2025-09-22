# MNIST Handwritten Digit Classification using Deep Learning (Neural Network)

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# print(X_test)
# print(X_train)

# scaling the values by dividing by values by 255 (the max value)
X_train = X_train/254
X_test = X_test/254

print(X_train)

# Building the Neural Network
# setting up the layers of the Neural Network

# model = keras.Sequential([
                        # keras.layers.Flattern(input_shape(28, 28)),   # for grayscale images, else we'll add 3 as the last parameter for RGB image
                        # keras.layers.Dense(50, activation='relu), 
                        # keras.layers.Dense(50, activation='relu), 
                        # keras.layers.Dense(10, activation='sigmoid), 
# ])

# how the layers works
    # the first layer takes the number of rows and column, then flatten it for all the values to be in a single column
    # 
    # the fourth layer is the output layer, the number of values (classes), for this project we have 10 classes


# compiling the Neural Network
#model.compile(optimizer = 'admin',
#             loss = 'sparse_categorical_crossentropy', # if the label encoding as on hot encoding (0 1 0 or 0  0 1) we use the 
              # above, else if it simple label enconding like (01234567) 
#             metrcis=['accuracy'])


# Training the Neural Network 

# Evaluate the model, by making Accuracy and Loss on the test data
# print(accuracy)

# print the evaluated model result
# plt.imshow(X_test[0])
# plt.show()
# print(Y_test[0])

# make predictions
# Y_pred = model.predict(X_test)

# print prediction image also print the label no
# print(Y_pred[0])

# --- argmax; returns the max value or number from the array list
# label_for_img = np.argmax(Y_pred[0])
# print(label_for_img)


# converting the prediction probabilities to class label for all test data points
# Y_pred_labels = [np.argmax(i) for i in Y_pred]
# print(Y_pred_labels)


# constructing the confusion matrix
# conf_mat = confusion_matrix(Y_test, Y_pred_labels)
# print(conf_mat)

# display the conf_mat in a graph

# - - - Building a predictive system - - -
# input_img = cv2.imread('dog.jpg')

# check the shape if it's RGB or Graysacle
# if RGB, convert to Grayscale
# grayscale = cv2.cvtcolor(input_img, cv2.COLOR_RGB2GRAY) 

# resize the grayscale image, then print
# input_img_resize = 

# division by 255

# reshaped the input img to what model expect, 
# Not: if it's RGB = np.reshape(input_img_resize, [1,28,28,3])
# else = np.reshape(input_img_resize, [1,28,28])

# Predict the image_reshaped, then print result 
# get the label prediction, then print

