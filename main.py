from flask import Flask, render_template, request, url_for
from keras.models import model_from_json
import computer_vision
import time
import os

import math
def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

UPLOAD_FOLDER = '/uploaded_images'
ALLOWED_EXTENSION = {'png', 'jpg', 'jpeg'}
loaded_model = ''
loaded_model_json = ''

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Here you write your routes and code


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        print("Post called")

    return render_template("home.html", somethingFromTheBackend="Your ai results")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 1. We get the images from the frontend
        image1 = request.files['image1']
        image2 = request.files['image2']
        image3 = request.files['image3']
        # same for the other 2 images

        # 2. Send the images to the ai function
        result1 = computer_vision.predict_image(image1.filename, loaded_model, loaded_model_json)
        result2 = computer_vision.predict_image(image2.filename, loaded_model, loaded_model_json)
        result3 = computer_vision.predict_image(image3.filename, loaded_model, loaded_model_json)

        # Checking the filenames of the images we uploaded
        print(image1.filename)
        print(image2.filename)
        print(image3.filename)

        result = [result1, result2, result3]

        print(result)

        if result[0] == 'times' or result[0] == '+' or result[0] == '-'  or result[0] == 'by' or result[0] == '=':
            result[0] = 5

        if result[2] == 'times' or result[2] == '+' or result[2] == '-'  or result[2] == 'by' or result[2] == '=':
            result[2] = 1    

        if result[1] == 'times':
            end_result = result[0] * result[2]
        elif result[1] == '+':
            end_result = result[0] + result[2]
        elif result[1] == '-':
            end_result = result[0] - result[2]
        elif result[1] == 'by':
            end_result = result[0] / result[2]  
        else:
            end_result = 'No result.'
        return render_template("result.html", aiResult=[result, str(truncate(end_result, 2))])

@app.route('/contact')
def contact():
    return 'This is the contact page'


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        return "Hello, " + username + " and your  password " + password

    return render_template("login.html")


# ---------------------

if __name__ == '__main__':
    # load json and create model
    json_file = open('new_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("new_model.h5")
    print("Loaded model from disk")
    app.run()
