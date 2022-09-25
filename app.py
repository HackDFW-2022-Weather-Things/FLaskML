from predict import predict
from http.server import SimpleHTTPRequestHandler, HTTPServer
from flask import Flask, request
import numpy as np
import base64
from PIL import Image
import io
import tensorflow as tf
import requests

app = Flask(__name__)

def get_model():
    return tf.keras.models.load_model("model.h5")

model = get_model()
@app.route("/testing", methods=['POST'])
def test_connection():
    return {"success": True}

@app.route("/prediction", methods=['POST'])
def get_prediction():
    photo = request.get_json()['photo']
    x = request.get_json()['x']
    y = request.get_json()['y']
    imgdata = base64.b64decode(photo)
    img = Image.open(io.BytesIO(imgdata))
    img = np.array(img) 
    img = img[:, :, ::-1].copy() 
    result = 0
    try:
        result = predict(img, model)
    except:
        result = -1
    # Send result to express
    myobj = {'data': {"x": x, "y": y, "result": result}}
    r = requests.post('http://54.161.43.254/publish', json = myobj)

    return {"result": result, "blob": r.text}

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('localhost', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    run()