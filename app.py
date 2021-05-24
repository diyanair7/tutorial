from flask import Flask, render_template, request
import keras
from keras.models import load_model
import cv2
import numpy as np


app = Flask(__name__)

model = load_model('model.h5')
print("model is loaded")
data=" "

@app.route('/')
def index():
    return render_template("index.html")



@app.route("/prediction", methods=["post"])
def prediction():

    img = request.files['img']
    img.save("img.jpeg")

    img3 = cv2.imread("img.jpeg")
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img3, (224,224))
    img3 = np.reshape(img3,(1,224,224,3))
    output = model.predict(img3)

    if output[0][0] > output[0][1]:
        a="Mammooty"
    else:
        a= "Mohanlal"

    return render_template("index.html", data=a)

if __name__ == "__main__":
    app.run(debug=True)
