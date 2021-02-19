# **Group project: Machine Learning with Python - Digit Recognition**

**Group members** <br/>
Emmmanuel Gogow <br/>
Mario Blauensteiner <br/>
Wilma Mikschl <br/>
David Riegger <br/>
Arpad Gerber <br/>


**About** <br/>
This is a student project of the University of St. Gallen of the course Programming with Advanced Computer Languages. <br/>
The goal of the project was to create a supervised  machine learning classifier which recognises hand written digits from 0 to 9.
The classifier is based on training and test data sets provided by the MNIST data base. We enhanced the program by adding a feature that allows the user to feed the classifier with images of self-written digits. The program output is the final digit classification.


**Pre-requisites** <br/>
The program is coded in Python3. The following libraries need to be installed prior to running the program: <br/>
*numpy, tensorflow, keras, sklearn, PIL, matplotlib*

**Instructions** <br/>
1. Prior to starting the programm, you might want to add some additional self-written images of digits to the folder 'Digit Images' (images should be in PNG or JPEG format).
2. Run the file Project_version_4.ipynb.
3. Choose which digit you want to classify by changing the file name of the respective image in the image converter.
4. The output should give the digit classification as well as a plot showing the recognition probabilites.

**Description** <br/>
First, the data was prepared. We load data sets for training and testing from the MNIST package to get the data for the digits 0 to 9. Then we downloaded a dataset with mathematical symbols from Kaggle to get the symbols +, -, /, *. The kaggle data had to be imported from a desktop folder and transformed to match the style of the MNIST data. After that, the digits and the mathematical symbols were merged to one dataset. Then the data was brought into a form that is optimized for the learning model and the values were normalized values to the range from 0 to 1. <br/>
Second, we build a sequential model based on a convolutional neural network using *tensorflow* and *keras*. <br/>
Thirdly, the established model was fitted to the training data. The model was fitted in 10 epochs and with a validation set that would indicate the out of sample performance of each epoch. The final model was found to have an accuracy of 99 % which seems to be unusually high. After investigating the Kaggle data and reading comments about the data set the concern arose that the majority of the images in the data set might be duplicates. Due to time constraints, we were not able to further investigate or fix the problem by deleting duplicates. However, the model was still able to classify our own handwriting. <br/>
Next, we import an image file of our handwritten digits. The image is then resized, centered, and the colors are inverted as the training data are images of white digits on black ground. After that, we convert the image into the format that our classifier model is based on. <br/>
Finally, we let the model predict the digit of the input image. This program was then integrated in our web app which was built by flask. The final result is the result of the calculation which is composed of three hand written operators that were uploaded to the webpage. 

**Sources** <br/>
data sets: <br/>
http://yann.lecun.com/exdb/mnist/tutorial <br/>
https://www.kaggle.com/hojjatk/read-mnist-dataset<br/>
https://www.kaggle.com/xainano/handwrittenmathsymbols<br/>
*tensorflow* tutorial: <br/>
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/#:~:text=MNIST%20Handwritten%20Digit%20Classification%20Dataset,-The%20MNIST%20dataset&text=It%20is%20a%20dataset%20of,from%200%20to%209%2C%20inclusively <br/>
*keras* tutorial: <br/>
https://www.datacamp.com/community/tutorials/deep-learning-python <br/>
https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical<br/>
https://www.tensorflow.org/tutorials/keras/classification<br/>
https://stackoverflow.com/questions/36952763/how-to-return-history-of-validation-loss-in-keras <br/>
image conversion: <br/>
https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/ <br/>
https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays<br/>
https://stackoverflow.com/questions/45826184/what-image-format-are-mnist-images<br/>
https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/<br/>
https://stackoverflow.com/questions/35842274/convert-own-image-to-mnists-image<br/>
https://stackoverflow.com/questions/42353676/display-mnist-image-using-matplotlib
*flask* turorial: <br/>
https://www.youtube.com/watch?v=Z1RJmh_OqeA
<br/>
save model and weights and reload the model and weights: </br>
https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018</br>
