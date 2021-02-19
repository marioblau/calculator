#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:14:26 2021

@author: gerberarpad
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2 # cv2
from tensorflow import keras
from tensorflow.keras import layers, Dense, Input, InputLayer, Flatten # Dense
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD
from PIL import Image, ImageFilter
from keras.datasets import mnist
from keras.models import model_from_json
import PIL.ImageOps
from PIL import Image, ImageFilter, ImageDraw


# get the images from folders on the desktop to the correct format
#==================================================================
# define the path to the images
img_folder = '/Users/gerberarpad/Desktop/Master/2.Sem/Winterschool/Project/archive/extracted_images-1'

# loop through all folders with symbols and loop through all their pictures to convert them 
# to the right format (numpy array). Then append them to a list which will be further processed.
# also append the names of the folders to a list, to use them as labels.
img_data_array=[]
class_name=[]

for dir1 in os.listdir(img_folder):
    
    print(dir1)
    
    if dir1 != '.DS_Store': # exclude ".DS_Store" because this is a default file 
                            #created by macOS in every folder. It is hidden but 
                            #causes problems in the loop.
           
        for file in os.listdir(os.path.join(img_folder, dir1)):
    
             #print(file)
             #print(dir1)
             if file != '.DS_Store':
             
                
                 image_path= os.path.join(img_folder, dir1,  file)
                
                 #print(image_path)
                 #print(image_path)
                 image= Image.open( image_path)
                 #print('ok')
                 image = image.convert('L')
                 image = image.resize((28,28), Image.LANCZOS) 
                 image = image.filter(ImageFilter.SHARPEN)
                 image = PIL.ImageOps.invert(image)
                 image = np.array(image, dtype='float32')
                 #plt.imshow(image, cmap='gray')
                 image = image / 255.0 
                 image = image.reshape((28,28,1))
                 img_data_array.append(image)
                 class_name.append(dir1)
                 
                 
# print one of the images to see how it looks like.
# you can see that it is color inverted, just like the mnist dataset
#==========================================  
pic = img_data_array[0].reshape((28, 28))
plt.imshow(pic, cmap='gray')
plt.show()

# convert the labels to numbers for further processing
labels = class_name
for i in range(len(labels)):
    if labels[i] == '+':
        labels[i] = 10
    if labels[i] == '-':
        labels[i] = 11
    if labels[i] == 'times':
        labels[i] = 12
    if labels[i] == 'by':
        labels[i] = 13
    if labels[i] == '=':
        labels[i] = 14

# we will use this function on the 'labels' list to see if all the 
# labels we want have been extracted.
def unique(list1):
     
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    for x in unique_list:
        print(x)
        
unique(labels)


# load mnist numbers
#==================================================================
# the mnist data set contains digits from 0 to 9.
# The quality of the images are much better than the ones
# in our symbols data set. So we will combine the two data sets 
# to have the best possible quality of digits and symbols.

(trainX, trainY_number), (testX, testY_number) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
#print(unique(trainY_number))
#trainY_number = to_categorical(trainY)
#testY_number = to_categorical(testY)
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')

# normalize the data to a range of 0-1 by dividing through maximum value
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0
trainX_number = train_norm
testX_number = test_norm



#add symbols to mnist numbers and combine the data from the folder 
#with the images from the mnist package
#======================================================

from sklearn.model_selection import train_test_split

x = np.array(img_data_array)
trainX_symbols, testX_symbols, trainY_symbols, testY_symbols = train_test_split(x, labels, test_size=0.20, random_state=42)

# Create an array with labels 0 1 2 3 4 5 6 7 8 9 (10(-) 11(+) 12(times) 13(by) 14(=)) for training
trainY = trainY_symbols
for i in trainY_number:
  trainY.append(i)

# Create an array with labels 0 1 2 3 4 5 6 7 8 9 (10(-) 11(+) 12(times) 13(by) 14(=)) for testing
testY = testY_symbols
for i in testY_number:
  testY.append(i)

# convert labels in array to categorical variables
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# merge the training and testing variables from the numbers data set and the symbols data set
trainX = np.concatenate((trainX_symbols, trainX_number)) 
testX = np.concatenate((testX_symbols, testX_number))


# merge the training and testing labels from the numbers data set and the symbols data set
# and merge the training and testing variables from the numbers data set and the symbols data set
Y = np.concatenate((trainY, testY)) # total Y

X = np.concatenate((trainX, testX)) # total X


# split the combined mnist and folder data to the final training and testing data set
#======================================================================================
trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.35, random_state=42)

print(trainX.shape)
print(testX.shape)

print(trainY.shape)
print(testY.shape)

#create a convolutional neural network
#======================================================
new_model = Sequential()
new_model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
new_model.add(MaxPooling2D((2, 2)))
new_model.add(Flatten())
new_model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
new_model.add(Dense(15, activation='softmax'))
new_model.summary()

#compile the model
#======================================================
opt = SGD(lr=0.01, momentum=0.9)
new_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model to the training data
#======================================================
history = new_model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(valX, valY), shuffle = False)



# save the model and its weights to a json file for further use.
#======================================================
# serialize to JSON
os.getcwd()
os.chdir('/Users/gerberarpad/Desktop/Master/2.Sem/Winterschool/Project')
new_model_json = new_model.to_json()
with open('new_model.json', "w") as file:
   file.write(new_model_json)
# serialize weights to HDF5
new_model.save_weights('new_model.h5')
print("Saved model to disk")



# make some pretty plots of the model's training progress 
# and insample performance
#================================================================
# At first the history is a keras history object which we convert
# to a pandas data frame in order to be able to plot the
# developpment of the training loss and the training classification accuracy.
print(type(history))
print(history.history)
hist=pd.DataFrame(data=history.history, dtype='float32')

# plot accuracy
# it is observable how the classification accuracy in the trainingset
# increases with every iteration to almost 100%
plt.title('Training Classification Accuracy')
plt.plot(hist['accuracy'], color='blue', label='train')
plt.show()

# plot loss
# similarly our loss decreases with every iteration of the neural network
plt.title('Training Loss')
plt.plot(hist['loss'], color='blue', label='train')


# evaluate the trained model based on the test accuracy for the test dataset i.e.
# out of sample performance
#===============================================================================
test_loss, test_acc =new_model.evaluate(testX, testY, verbose=2)
print('\nTest accuracy:', test_acc)
# Test accuracy: 0.9937571883201599

# load saved model
#====================================

# later to reload the trained model
 
# load json and create model
json_file = open('new_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("new_model.h5")
print("Loaded model from disk")





# evaluate loaded model on test data
#========================================================

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
loaded_test_loss, loaded_test_acc = loaded_model.evaluate(testX, testY, verbose=2)
print('\nLoaded_Test accuracy:', loaded_test_acc)
# Loaded_Test accuracy: 0.9937571883201599


# Predict image functiom
#==========================================================================
# build an image converter with the aim that the model predicts the digit of the new uploaded image 
# open the image, if saved in the same dictionary the name suffices, otherwise use whole savepath

def predict_image(imagetopredict):

  
  im = Image.open(imagetopredict)

  # get the bands in the image since python imaging library allows to store several band in a single image
  im.getbands() 

  # define type and depth of a pixel in the image
  # use mode L: 8-bit pixels (has a range of 0-255), black and white
  im = im.convert('L') 
  im.getbands()

  # resizing and quality enhancement of the image
  im = im.resize((28,28), Image.LANCZOS) 
  im = im.filter(ImageFilter.SHARPEN) 

  # convert to a numpy array
  test = np.array(im, dtype='float32')

  # examination of the image format through visualization
  # plot will appear at the end of the output
  plt.imshow(test, cmap='gray') 

  # insight: Color of the image has to be inverted because in the MNIST dataset they are written in white on a black canvas
  # insight: Therefore the present digit needs to be color inverted
  

  # invert the colors of the image (change black and white) 
  # visualization: illustration of the shape to be recognized by the model
  inverted_image = PIL.ImageOps.invert(im)
  # convert the image to a numpy array
  inverted_image = np.array(inverted_image, dtype='float32')
  converted_image = inverted_image.reshape((28, 28))
  plt.imshow(converted_image, cmap='gray')
  plt.show()

  # scale the color values of the pixels in the image down to values between 0 - 1 by dividing 
  # each of them by the maximum color value of 255
  inverted_image = inverted_image / 255.0 

  # reshape the image in order to meet the form requirements for the neural network
  inverted_image = inverted_image.reshape((1,28,28,1))

  # predict the digit of our own image with the trained and tested model
  # output with highest probability corresponds to the predicted digit by the model
  predictions = new_model.predict(inverted_image)

  print(np.argmax(predictions))

  if np.argmax(predictions) > 0 and np.argmax(predictions) < 10:
    perdicted_value = np.argmax(predictions)
  elif np.argmax(predictions) == 14:
    perdicted_value = '='
  elif np.argmax(predictions) == 13:
    perdicted_value = 'by'
  elif np.argmax(predictions) == 12:
    perdicted_value = 'times'
  elif np.argmax(predictions) == 11:
    perdicted_value = '-'
  elif np.argmax(predictions) == 10:
    perdicted_value = '+'
  else:
    perdicted_value = 'undefined'


  print("The hand written sign was classified as the digit :", perdicted_value)


  # visualization of the predicted probability for each digit
  plt.bar(np.arange(len(predictions[0])),predictions[0])
  plt.xticks(np.arange(len(predictions[0])))
  plt.ylim(0, 1.1)
  plt.hlines(1,0,9,color='gray',linestyles='dotted')
  plt.title("Prediction probability")
  plt.xlabel("Predicted digit")
  plt.show()
  # the plot below indicates with which probability the input can be assigned to each class of digits.

  return perdicted_value

# predict_image('/Users/gerberarpad/Desktop/Master/2.Sem/Winterschool/Project/archive/extracted_images-1/test/IMG+.jpeg')










