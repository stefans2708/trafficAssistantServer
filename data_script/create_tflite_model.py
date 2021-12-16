import os

import tensorflow as tf
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

MODEL_DIR_PATH = 'D:\\Development\\ObjectDetection\\training\\tflite\\saved_model'
DESTINATION = 'D:\\Development\\ObjectDetection\\training\\tflite\\'
MODEL_QUANITIZED_WEIGHTS_FILE = 'quanitized.tflite'
MODEL_QUANITIZED_FILE = 'fully_quanitized_model.tflite'
MODEL_FILE = 'lite_model.tflite'
LABELS_FILE = 'D:\\Development\\ObjectDetection\\training\\labels.txt'
IMAGES_FOLDER_PATH = '/home/stefan/Documents/ObjectDetection/Images/generated_320x240_representative_dataset'
_INPUT_NORM_MEAN = 127.5
_INPUT_NORM_STD = 127.5

""" Regular tflite model """
tflite_model_path = os.path.join(DESTINATION, MODEL_FILE)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR_PATH)
open(tflite_model_path, "wb").write(converter.convert())

# noinspection PyTypeChecker
writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(tflite_model_path), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [LABELS_FILE]
)
writer_utils.save_file(writer.populate(), tflite_model_path)

""" Quanitized tflite model """
tflite_model_path = os.path.join(DESTINATION, MODEL_QUANITIZED_WEIGHTS_FILE)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
open(tflite_model_path, "wb").write(converter.convert())

# noinspection PyTypeChecker
writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(tflite_model_path), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [LABELS_FILE]
)
writer_utils.save_file(writer.populate(), tflite_model_path)

# Verify the populated metadata and associated files.
# displayer = metadata.MetadataDisplayer.with_model_file(tflite_model_path)
# print("Metadata populated:")
# print(displayer.get_metadata_json())
# print("Associated file(s) populated:")
# print(displayer.get_packed_associated_file_list())
