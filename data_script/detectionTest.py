import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
from six import BytesIO

matplotlib.use('TkAgg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = 'D:\\Development\\ObjectDetection\\training\\model\\saved_model'
IMAGES_DIR = 'C:\\Users\\stefa\\Desktop\\test'
category_index = {
    1: {'id': 1, 'name': 'car'},
    2: {'id': 2, 'name': 'truck'},
    3: {'id': 3, 'name': 'bus'},
    4: {'id': 4, 'name': 'biker'},
    5: {'id': 5, 'name': 'pedestrian'}
}


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (img_width, img_height) = image.size
    return np.array(image.getdata()).reshape((img_height, img_width, 3)).astype(np.uint8)


if __name__ == "__main__":
    detect_fn = tf.saved_model.load(MODEL_PATH)
    elapsed = []
    count = len(os.listdir(IMAGES_DIR))
    for index, img in enumerate(os.listdir(IMAGES_DIR)):
        image_path = os.path.join(IMAGES_DIR, img)
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = np.expand_dims(image_np, 0)
        start_time = time.time()
        detections = detect_fn(input_tensor)
        end_time = time.time()
        elapsed.append(end_time - start_time)

        plt.rcParams['figure.figsize'] = [42, 21]
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.60,
            agnostic_mode=False)
        plt.subplot(1, count, index + 1)
        plt.imshow(image_np_with_detections)
    plt.show()
    mean_elapsed = sum(elapsed) / float(len(elapsed))
    print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
