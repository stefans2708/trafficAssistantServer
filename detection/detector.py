import time

import numpy as np
import tensorflow as tf

import utils
from detection.detection import Detection

class_names = ['car', 'truck', 'bus', 'biker', 'pedestrian']


class ObjectDetector:

    def __init__(self):
        self.model = tf.saved_model.load('model/detection')

    @staticmethod
    def get_class_name(model_result, detection_index):
        return class_names[model_result['detection_classes'][0][detection_index].numpy().astype(np.int32) - 1]

    @staticmethod
    def get_confidence(model_result, detection_index):
        return model_result['detection_scores'][0][detection_index].numpy() * 100

    @staticmethod
    def get_bounding_box(model_result, image_width, image_height, detection_index):
        box = model_result['detection_boxes'][0][detection_index].numpy()  # y1 x1 y2 x2
        return [box[1] * image_width, box[0] * image_height, box[3] * image_width, box[2] * image_height]

    def run_detection(self, image, confidence_threshold, max_detections):
        start_time = time.time()
        model_result = self.model(np.expand_dims(image, 0))  # model accepts only batches
        print('Elapsed time: ', time.time() - start_time)

        detections = []
        count = min(max_detections, int(model_result['num_detections'][0].numpy()))
        height, width, _ = image.shape
        for i in range(count):
            confidence = self.get_confidence(model_result, i)
            if confidence < confidence_threshold:
                continue

            detections.append(
                Detection(title=self.get_class_name(model_result, i),
                          confidence=confidence,
                          location=self.get_bounding_box(model_result, width, height, i)))

        return utils.array_to_json(detections)
