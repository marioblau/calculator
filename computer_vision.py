import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2 
from tensorflow import keras
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
import PIL.ImageOps
from PIL import Image, ImageFilter, ImageDraw

# Predict image functiom
#==========================================================================
# build an image converter with the aim that the model predicts the digit of the new uploaded image 
# open the image, if saved in the same dictionary the name suffices, otherwise use whole savepath

def predict_image(imagetopredict, loaded_model, loaded_model_json):
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

  # implementation of color inversion (now black = 0 and white = 255, mode L)
  
  #newImage = Image.new('L', (28, 28), (255)) 

  # implementation of image centralization (vertical and horizontal distance from upper left corner)
  
  #newImage.paste(im,(4,4)) 

  # invert the colors of the image (change black and white) 
  # visualization: illustration of the shape to be recognized by the model
  inverted_image = PIL.ImageOps.invert(im)
  # convert the image to a numpy array
  inverted_image = np.array(inverted_image, dtype='float32')
  converted_image = inverted_image.reshape((28, 28))
  # plt.imshow(converted_image, cmap='gray')
  # plt.show()

  # scale the color values of the pixels in the image down to values between 0 - 1 by dividing 
  # each of them by the maximum color value of 255
  inverted_image = inverted_image / 255.0 

  # reshape the image in order to meet the form requirements for the neural network
  inverted_image = inverted_image.reshape((1,28,28,1))

  # predict the digit of our own image with the trained and tested model
  # output with highest probability corresponds to the predicted digit by the model
  
  predictions = loaded_model.predict(inverted_image)

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
  return perdicted_value