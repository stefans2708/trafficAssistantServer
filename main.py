import base64

import cv2
import numpy as np
from flask import Flask, request, session
from flask_socketio import SocketIO

import constants
from constants import *
from detector import ObjectDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = constants.SECRET_KEY
socketio = SocketIO(app)
object_detector = ObjectDetector()


@app.route('/')
def index():
    return 'Hello, world!'


@socketio.event
def connect():
    confidence_threshold = int(request.args.get('confidenceThreshold'))
    session[SESSION_CONFIDENCE_THRESHOLD] = \
        DEFAULT_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold

    max_detections = int(request.args.get('maxDetections'))
    session[SESSION_MAX_DETECTIONS] = \
        DEFAULT_MAX_DETECTIONS if max_detections is None else max_detections

    print(f'New client ({request.sid}). '
          f'Settings: confidence threshold: {confidence_threshold}, max detections: {max_detections}')


@socketio.event
def disconnect():
    print(f'Client {request.sid} disconnected.')


@socketio.event
def image(data):
    image_bytes = base64.b64decode(data)
    np_data = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    frame = np.expand_dims(frame, 0)
    return object_detector.run_detection(image=frame,
                                         confidence_threshold=session[SESSION_CONFIDENCE_THRESHOLD],
                                         max_detections=session[SESSION_MAX_DETECTIONS])


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=9990, log_output=True, debug=False)
