# **Group project: Machine Learning with Python - Digit Recognition**

**Group members** <br/>
Emmmanuel Gogow <br/>
Mario Blauensteiner <br/>
Wilma Mikschl <br/>
David Riegger <br/>
Arpad Gerber <br/>
Bryan Rhyner <br/>


**About** <br/>
This is a project from students of the University of St. Gallen. This project was conducted as a part of the Winterschool 2021 which is a workshop hosted by the SHSG, the students association at the university of St. Gallen. <br/>
The goal of the project was to create a supervised machine learning classifier which recognises handwritten digits from 0 to 9 and the following mathematical operations: +, -, /, *. The classifier was then integrated in a web which enables users to upload a picture of a handwritten calculation (e.g., 7 + 7) and obtain a result (e.g., 14). The motivation for this calculator app was the fact that students often face the issue of low interpretability when looking at previously made notes, especially when the later were made under time pressure. 
The data used to train the classifier was obtained from two separate sources and then merged together. For the digits (0 to 9) the MNIST data set was used. The pictures of the mathematical operations were drawn from an open data set on Kaggle. The user can feed the classifier with images of self-written digits which can be uploaded on a webpage and will then be classified by the algorithm. The program output is the final result of the handwritten calculation.



**Pre-requisites** <br/>
The program is coded in Python3 and HTML. Visual Studio code was used to implement the web app with flask. The following libraries need to be installed prior to running the program: <br/>
*numpy, tensorflow, keras, sklearn, PIL (pillow), matplotlib, pandas, python flask*


**Instructions** <br/>
1. Prior to testing the program, you might want to prepare some additional self-written images of digits and mathematical operations (images should be in PNG or JPEG or JPG format). In any case you can use the images provided in the folder *test-images*.
2. Open the GitHub repository in visual studio code. Run main.py and then access the route called */uploader* on the development server which will bring you to the page where you will be presented with three file upload slots. 
3. Upload the digits and mathematical signs needed for your calculation and you will get the result.


**Description** <br/>
First, the data was prepared. We load data sets for training and testing from the MNIST package to get the data for the digits 0 to 9. Then we downloaded a dataset with mathematical symbols from Kaggle to get the symbols +, -, /, *. The kaggle data had to be imported from a desktop folder and transformed to match the style of the MNIST data. After that, the digits and the mathematical symbols were merged to one dataset. Then the data was brought into a form that is optimized for the learning model and the values were normalized values to the range from 0 to 1. <br/>
Second, we build a sequential model based on a convolutional neural network using *tensorflow* and *keras*. <br/>
Thirdly, the established model was fitted to the training data. The model was fitted in 10 epochs and with a validation set that would indicate the out of sample performance of each epoch. <br/>
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
https://stackoverflow.com/questions/42353676/display-mnist-image-using-matplotlib<br/>
*flask* turorial: <br/>
https://www.youtube.com/watch?v=Z1RJmh_OqeA
<br/>
save model and weights and reload the model and weights: </br>
https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018</br>
