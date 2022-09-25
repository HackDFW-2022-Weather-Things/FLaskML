from predict import predict
from http.server import SimpleHTTPRequestHandler, HTTPServer
from flask import Flask, request
import numpy as np
import base64
from PIL import Image
import io
import json
import uuid
import calendar
import time
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


def putDynamoDB(longitude_x,latitude_y):
    my_config = Config(region_name = 'us-east-1')
    dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id= os.getenv('ACCESS_KEY'),
    aws_secret_access_key= os.getenv('SECRET_KEY'),
    config = my_config)

    table = dynamodb.Table('express-app')

    GMT = time.gmtime()
    keyUUID = uuid.uuid4()
    uuidStr = str(keyUUID)

    putID = uuidStr
    putX = longitude_x
    putY = latitude_y
    putSeverity = 'severe'
    putTimestamp = calendar.timegm(GMT) 

    try:
        response = table.put_item(
        Item={
              'id': putID,
              'longitude_x': putX,
              'latitude_y': putY,
              'severity': putSeverity,
              'timestamp': putTimestamp
        }
        )
    except:
        response =-1
    return response

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
   # Send results to DynamoDB
    putDynamoDB(x,y)
    r = requests.post('http://54.161.43.254/publish', json = myobj)

    return {"blob": r.text}

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('localhost', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    run()