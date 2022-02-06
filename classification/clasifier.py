import time

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

import utils
from classification.classification import Classification


class Classifier:

    def __init__(self):
        self.model = tf.keras.models.load_model('../model/classification/classification_model.h5')
        self.labels = self.init_labels(metadata_file='../model/classification/cars_meta.mat')

    @staticmethod
    def init_labels(metadata_file):
        metadata = loadmat(metadata_file)
        labels = list()
        for lbl in metadata['class_names'][0]:
            labels.append(lbl[0])
        return labels

    def classify(self, image, classification_threshold):
        start_time = time.time()
        predictions = self.model.predict(image)
        print('Elapsed time: ', time.time() - start_time)

        rounded_predictions = np.round(predictions, decimals=4)
        prediction_pairs = list(enumerate(rounded_predictions[0]))
        prediction_pairs.sort(key=lambda pair: pair[1], reverse=True)  # [(17,0.92), (4, 0.08), (1, 00) ... (196, 0)]

        classifications = []
        confidence = 100
        index = 0
        while index < len(self.labels) and confidence >= classification_threshold:
            confidence = [prediction_pairs[index][1]]
            classifications.append(
                Classification(
                    title=self.labels[prediction_pairs[index][0]],
                    confidence=confidence
                )
            )
            index += 1

        return utils.array_to_json(classifications)