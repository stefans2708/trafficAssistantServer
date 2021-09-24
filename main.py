import base64

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, session
from flask_socketio import SocketIO

import constants
from object_detector import run_detection, as_json_format
from session_settings import SessionSettings

app = Flask(__name__)
app.config['SECRET_KEY'] = 'this_is_secret_123'
socketio = SocketIO(app, cors_allowed_origins='*')
model = tf.saved_model.load('saved_model')
settings = SessionSettings()


@app.route('/')
def index():
    return 'Hello, world!'


@socketio.event()
def connect():
    print('Connected', request.sid)
    print(request.args.get('maxDetections'))
    print(request.args.get('confidenceThreshold'))
    print(request.args.get('imageWidth'))
    print(request.args.get('imageHeight'))


@socketio.on(constants.EVENT_MESSAGE)
def handle_message(message):
    print(f'{request.sid}: {message}')
    return f'OK, {message}'


@socketio.event()
def image(data):
    image_bytes = base64.b64decode(data)
    np_data = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (480, 640))
    frame = np.expand_dims(frame, 0)

    # with open ("received.jpg", "wb") as fh:
    #     fh.write(image_bytes)

    return as_json_format(run_detection(model, frame))


if __name__ == '__main__':
    # socketio.run(app, host='192.168.0.21', port=9990, debug=True)
    socketio.run(app, host='192.168.1.109', port=9990, debug=True)
