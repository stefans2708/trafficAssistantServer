"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
from collections import namedtuple

import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

IMAGES_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\training'
CSV_FILE_PATH = 'C:\\Users\\stefa\\Desktop\\training.csv'
OUTPUT_PATH = 'C:\\Users\\stefa\\Desktop\\training.record'


def class_text_to_int(row_label):
    return {
        'car': 1,
        'truck': 2,
        'bus': 3,
        'biker': 4,
        'pedestrian': 5
    }.get(row_label, None)


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    difficults = []
    truncateds = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        difficults.append(int(row['difficult']))
        truncateds.append(int(row['truncated']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficults),
        'image/object/truncated': dataset_util.int64_list_feature(truncateds)
    }))
    return tf_example


def main():
    writer = tf.io.TFRecordWriter(OUTPUT_PATH)
    path = os.path.join(IMAGES_DIRECTORY)
    examples = pd.read_csv(CSV_FILE_PATH)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(OUTPUT_PATH))


if __name__ == '__main__':
    main()
