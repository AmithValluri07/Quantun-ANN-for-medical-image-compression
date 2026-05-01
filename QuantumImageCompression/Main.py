import matplotlib.pyplot as plt
import io
import base64
import QANN  #loading Qiskit QANN model for quantum features extraction
from skimage.metrics import structural_similarity as ssim #function to calculate SSIM
import os
import cv2
import numpy as np
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input, Conv2D, UpSampling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from math import log10, sqrt
import keras
import warnings
from PIL import Image
import matplotlib
matplotlib.use("Agg")
#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory


app = Flask(__name__)
app.secret_key = 'welcome'

qann = QANN.loadQANN()

def getModel():
    input_img = Input(shape=(128, 128, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    qann_input = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(qann_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    qann_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    qann_model = Model(input_img, qann_output)
    qann_model.compile(optimizer='adam', loss='mean_squared_error')
    qann_model.load_weights("model/qann_weights.hdf5")
    return qann_model

#call function to generate high quality image from low quality
def getSuperImage(path):
    img = Image.open(path) #load image from given path
    img = np.array(img)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("model/FSRCNN_x3.pb") 
    sr.setModel("fsrcnn",3)
    result = sr.upsample(img)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

def decodeCompressImage(path):
    qann_model = getModel()
    original = cv2.imread(path) #reading original image
    original = cv2.resize(original, (128, 128), cv2.INTER_LANCZOS4)
    test = QANN.getQANNFeatures(path, qann)#get QANN features
    temp = []
    temp.append(test)
    test = np.asarray(temp)
    test = test.astype('float32')
    test = test/255
    predict = qann_model.predict(test)#applying qann classical model on qann features to decompress image 
    predict = predict[0]#get predicted decompress image
    super_image = getSuperImage("test.jpg")
    figure, axis = plt.subplots(nrows=1, ncols=3,figsize=(8,4))#visualize images
    axis[0].set_title("Original Image")
    axis[1].set_title("QANN Decompressed Image")
    axis[2].set_title("Extension Super Enhanced Image")
    axis[0].imshow(original)
    axis[1].imshow(predict)
    axis[2].imshow(super_image)
    figure.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.clf()
    plt.cla()
    return img_b64

@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/UserLogin', methods=['GET', 'POST'])
def UserLogin():
    return render_template('UserLogin.html', msg='')

@app.route('/UserLoginAction', methods=['GET', 'POST'])
def UserLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('UserScreen.html', msg="<font size=3 color=blue>Welcome "+user+"</font>")
        else:
            return render_template('UserLogin.html', msg="<font size=3 color=red>Invalid login details</font>")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        data = request.files['t1']
        data = data.read()
        if os.path.exists("static/test.jpg"):
            os.remove("static/test.jpg")
        with open("static/test.jpg", "wb") as file:
            file.write(data)
        file.close()
        img_b64 = decodeCompressImage("static/test.jpg")
        return render_template('Predict.html', msg='', img = img_b64)

if __name__ == '__main__':
    app.run()    
