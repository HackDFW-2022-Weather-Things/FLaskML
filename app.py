from predict import predict
from http.server import SimpleHTTPRequestHandler, HTTPServer
from flask import Flask, request
import numpy as np
import base64
from PIL import Image
import io
import json
import tensorflow as tf
import requests
from distutils.command.config import config
import boto3
import os
import tqdm
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def get_model():
    return tf.keras.models.load_model("modelBig.h5")


def putDynamoDB():
    my_config = Config(region_name = 'us-east-1')
    dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id= os.getenv('ACCESS_KEY'),
    aws_secret_access_key= os.getenv('SECRET_KEY'),
    config = my_config)

    table = dynamodb.Table('express-app')

    table.put_item(
        Item={
              'id': '8000',
              'longitude_x': '32.7878937',
              'latitude_y': '46.7996563',
              'severity': 'calm',
              'timestamp': '1664097931'
        }
        )
    return 0

model = get_model()
model.summary()
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
    img = img / 255.0
    img = img[:, :, ::-1].copy() 
    result = 0
    print("made it to the try")
    try:
        result = predict(img, model)
    except:
        result = -1
    # Send result to express
    myobj = {'data': {"x": x, "y": y, "result": json.dumps(result.tolist())}}
    #botoObj ={'Item':{"id": 0, "longitude_x": x, "latitude_y": y, "severity": 'severe', 'timestamp': 123232332 }}
    putDynamoDB()
    r = requests.post('http://54.161.43.254/publish', json = myobj)

    return {"blob": r.text}

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('localhost', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    run()