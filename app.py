

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import cv2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import pickle
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH ='new_vgg19_20sep.h5'


# Load your trained model
# model = load_model(MODEL_PATH)



model1 = pickle.load(open("RandomForestClassifier_10.pkl", 'rb'))





# def model_predict(img_path, model):
#     image = cv2.imread(img_path,1)

    
#     img = cv2.resize(image, (224,224),
#                interpolation = cv2.COLOR_RGB2BGR)
   
#     x = img.reshape(1,224,224,3)
   

  

#     preds = model.predict(x)
#     preds=np.argmax(preds, axis=1)
#     if preds==0:
#         preds="The Person is Infected With glioma "
#     elif preds == 1:
#         preds="The Person Has NO_tumor"
#     elif preds == 2:
#         preds = "The person is Menginioun"
#     elif preds ==3:
#         preds = "The person is Pituratry"
    
    
#     return preds

@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')

@app.route('/about')
def analysis():
    # Main page
    return render_template('about.html')

@app.route('/heart')
def heart():
    # Main page
    return render_template('heart.html')

@app.route('/index', methods=['GET'])
def index():
    # Main page
    return render_template('brain.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path, model)
#         result=preds
#         return result
#     return None


@app.route('/result',methods=['GET','POST'])
def home1():
    if request.method =="POST":
        gender = request.form['gender']
        age = request.form['age']
        marital = request.form['marital']
        ethanicity = request.form['ethanicity']
        bmi = request.form['bmi']
        smoke = request.form['smoke']
        sports = request.form['sports']
        health = request.form['health']
        walk = request.form['walk']
        pdays = int(request.form['pdays'])
        stroke = request.form['stroke']
        attack = request.form['attack']
        kidney = request.form['kidney']
        other_cancer = request.form['other_cancer']
        lungs = request.form['lungs']

        bmi1=0
        smoke1 =0
        stroke1 =0 
        gender1= 0
        attack1 =0 
        age1 = 0
        ethanicity1 = 0 
        sports1 = 0 
        health1 = 0
        kidney1 = 0
        other_cancer1 = 0
        lungs1 = 0
        marital1 = 0
        walk1=0

        if gender == 'Male-1':
            gender1 =1
        elif gender == 'Female-2':
            gender1 = 2
        
        if age == '18 <= AGE <= 24 --- 1':
            age1 = 1
        elif age == '25 <= AGE <= 34 --- 2':
            age1 = 2
        elif age == '35 <= AGE <= 44 --- 3':
            age1 = 3
        elif age == '45 <= AGE <= 54 --- 4':
            age1 = 4
        elif age == '55 <= AGE <= 64 --- 5':
            age1 = 5
        elif age == '64 <= AGE  --- 5':
            age1 = 6

        if marital == 'Married-1':
            marital1 = 1
        elif marital == 'Divorced-2':
            marital1 = 2
        elif marital == 'Widowed-3':
            marital1 = 3
        elif marital == 'Separated-4':
            marital1 = 4
        elif marital == 'Never Married-5':
            marital1 = 5
        elif marital == 'A member of an unmarried couple-6':
            marital1 = 6

        if ethanicity == 'white-1':
            ethanicity1 = 1
        elif ethanicity == 'Black-2':
            ethanicity1 = 2
        elif ethanicity == 'Asian-3':
            ethanicity1 = 3
        elif ethanicity == 'American Indian-4':
            ethanicity1 = 4
        elif ethanicity == 'Hispanic--5':
            ethanicity1 = 5
        elif ethanicity == 'others--6':
            ethanicity1 = 6

        if bmi == '1-UnderWeight BMI':
            bmi1 = 1.0
        elif bmi == '2-Normal Weight':
            bmi1 = 2.0
        elif bmi == '3-OverWeight':
            bmi1 = 3.0
        elif bmi == '4-OverWeight':
            bmi1 = 4.0

        if smoke == '1-Smoke Every Day':
            smoke1 = 1
        elif smoke == '2-Smoke Some day':
            smoke1 = 2
        elif smoke == '3-Former Smoker':
            smoke1 = 3
        elif smoke == '4-Never Smoked':
            smoke1 = 4

        if sports == 'Yes-1':
            sports1 =1
        elif sports == 'No-2':
            sports1 =2

        if health == 'Excellent-1':
            health1 = 1
        elif health == 'Very Good-2':
            health1 = 2
        elif health == 'Good-3':
            health1 = 3
        elif health == 'Fair-4':
            health1 = 4
        elif health == 'Poor-5':
            health1 = 5
        if walk == 'Yes-1':
            walk1 = 1
        elif walk == 'No-2':
            walk1 = 2

        #<!-- _PHYS14D -->

        if stroke == 'Yes-1':
            stroke1= 1
        elif stroke == 'No-2':
            stroke1 = 2
			
		
        if attack == 'Yes-1':
            attack1 = 1
        elif attack == 'No-2':
            attack1 = 2
        if kidney =='Yes-1':
            kidney1 = 1

        elif kidney == 'No-2':
            kidney1 = 2

        if other_cancer == 'Yes-1':
            other_cancer1 = 1
        elif other_cancer == 'No-2':
            other_cancer1 = 2
		
        if lungs == 'Yes-1':
            lungs1 =1
        elif lungs == 'No-2':
            lungs1 = 2

        print(bmi,smoke,stroke ,gender,attack,age,ethanicity,sports,health,kidney,other_cancer,lungs,marital,walk,pdays)

        print(bmi1,smoke1 ,stroke1 ,gender1,attack1,age1,ethanicity1,sports1,health1,kidney1,other_cancer1,lungs1,marital1,walk1,pdays)

        predict = model1.predict([[bmi1,smoke1 ,stroke1 ,gender1,attack1,age1,ethanicity1,sports1,health1,kidney1,pdays,other_cancer1,lungs1,marital1,walk1]])
        #predict = model1.predict([[1.0,1,2.0,2,2.0,5,1,1.0,2.0,2.0,2,1.0,1.0,2.0,2.0]])
         
        if predict == 0 :
            prediction = "Good"
        else:
            prediction = "Not Good"
        return render_template("result.html", prediction_text="Heart status is --> {}".format(prediction)) 

        


    else:
        render_template('home.html')





if __name__ == '__main__':
    app.run(debug=True)
