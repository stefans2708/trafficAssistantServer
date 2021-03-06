from flask import Flask, request, session
from flask_socketio import SocketIO

import constants
import utils
from classification.clasifier import Classifier
from constants import *
from detection.detector import ObjectDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = constants.SECRET_KEY
socketio = SocketIO(app)
object_detector = ObjectDetector()
object_classifier = Classifier()


@app.route('/')
def index():
    return 'Hello, world!'


@socketio.event
def connect():
    detection_threshold = int(request.args.get('detectionThreshold'))
    session[SESSION_DETECTION_THRESHOLD] = \
        DEFAULT_CONFIDENCE_THRESHOLD if detection_threshold is None else detection_threshold

    classification_threshold = int(request.args.get('classificationThreshold'))
    session[SESSION_CLASSIFICATION_THRESHOLD] = \
        DEFAULT_CONFIDENCE_THRESHOLD if classification_threshold is None else classification_threshold

    max_detections = int(request.args.get('maxDetections'))
    session[SESSION_MAX_DETECTIONS] = \
        DEFAULT_MAX_DETECTIONS if max_detections is None else max_detections

    print(f'New client ({request.sid}). '
          f'Settings: detection threshold: {detection_threshold}, '
          f'classification threshold: {classification_threshold}, '
          f'max detections: {max_detections}')


@socketio.event
def disconnect():
    print(f'Client {request.sid} disconnected.')


@socketio.event
def image(data):
    return object_detector.run_detection(image=utils.image_from_base64_encoded_bytes(data),
                                         confidence_threshold=session[SESSION_DETECTION_THRESHOLD],
                                         max_detections=session[SESSION_MAX_DETECTIONS])


@socketio.event
def classify_cars(*data):
    classification_results_json = []
    for img in data:
        classification_results_json.append(
            object_classifier.classify(image=utils.image_from_base64_encoded_bytes(img),
                                       classification_threshold=session[SESSION_CLASSIFICATION_THRESHOLD]
                                       ).serialize_to_json()
        )
    result = "[" + ", ".join(classification_results_json) + "]"
    print(result)
    return result


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=9990, log_output=True, debug=False)
