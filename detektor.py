import os
import time

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from six import BytesIO

from detection import Detection

matplotlib.use('TkAgg')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = '/home/stefan/ObjectDetection/models/research/object_detection/training/colab/v7/exported_graph/saved_model'
class_names = [
    'children_ahead',
    'crosswalk',
    'keep_left',
    'keep_right',
    'left_curve_ahead',
    'no_entry',
    'no_overtaking',
    'no_parking',
    'no_traffic',
    'parking',
    'priority_road',
    'right_curve_ahead',
    'road_works',
    'speed_bump',
    'speed_limit_40',
    'speed_limit_50',
    'speed_limit_60',
    'speed_limit_80',
    'stop',
    'yield'
]


# TODO: remove later
def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (img_width, img_height) = image.size
    return np.array(image.getdata()).reshape((img_height, img_width, 3)).astype(np.uint8)


# TODO: read from settings
max_detections = 3
confidence_threshold = 40
image_size = (1280, 960)


def get_class_name(model_result, detection_index):
    return class_names[model_result['detection_classes'][0][detection_index].numpy().astype(np.int32) - 1]


def get_confidence(model_result, detection_index):
    return model_result['detection_scores'][0][detection_index].numpy() * 100


def get_bounding_box(model_result, detection_index):
    # y1 x1 y2 x2
    box = model_result['detection_boxes'][0][detection_index].numpy()
    return [box[1] * image_size[0], box[0] * image_size[1], box[3] * image_size[0], box[2] * image_size[1]]


def get_detections(model_result):
    detections = []
    count = min(max_detections, int(model_result['num_detections'][0].numpy()))
    for i in range(count):
        detections.append(
            Detection(clazz=get_class_name(model_result, i),
                      confidence=get_confidence(model_result, i),
                      bbox=get_bounding_box(model_result, i)))
    return detections


def run(model, image):
    start_time = time.time()
    detections = model(image)
    elapsed = time.time() - start_time
    print('Elapsed time: ', elapsed)

    return get_detections(detections)


if __name__ == "__main__":
    detector_model = tf.saved_model.load(MODEL_PATH)
    image_path = '/home/stefan/Documents/ObjectDetection/Images/testing/test_single/1277106794Image000036.jpg'
    img = load_image_into_numpy_array(image_path)
    img = np.expand_dims(img, 0)
    run(detector_model, img)
