import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as image

from scipy.io import loadmat

# force CPU execution to reduce init time
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PATH = 'C:\\Users\\stefa\\Desktop\\cars196'
IMAGE_SIZE = (192, 192)

meta = loadmat(os.path.join(PATH, 'devkit\\cars_meta.mat'))

labels = list()
for lbl in meta['class_names'][0]:
    labels.append(lbl[0])


def get_test_image_path(file):
    return PATH + '\\test\\' + file + '.jpg'


def display_predictions(predictions_ind_acc):
    plt_index = 9
    for index, accuracy in predictions_ind_acc:
        if accuracy < 0.05:
            break
        plt.subplot(4, 4, plt_index)
        plt_index += 1
        plt.axis("off")
        plt.title("Class: {}\naccuracy: {}".format(labels[index], round(accuracy * 100, ndigits=4)),
                  fontdict={'fontsize': 10})


testImage = image.load_img(path=get_test_image_path('Volkswagen-Golf_II_GTI1'), target_size=IMAGE_SIZE,
                           interpolation='bilinear')
input_arr = image.img_to_array(testImage)
input_arr = np.array([input_arr])

model = tf.keras.models.load_model('../model/classification/classification_model.h5')
predictions = model.predict(input_arr)

# predictions is list of confidence for each class of 196 cars. Classes are sorted by name (1..196)
print(predictions)
rounded_predictions = np.round(predictions, decimals=4)
prediction_pairs = list(enumerate(rounded_predictions[0]))
prediction_pairs.sort(key=lambda pair: pair[1], reverse=True)

#[(190, 0.9976), (156, 0.0009), (192, 0.0006), (28, 0.0002), (17, 1e-04), (25, 1e-04), (35, 1e-04), (81, 1e-04), (0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0),..., (195, 0.0)]
print(prediction_pairs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.imshow(testImage)
plt.axis("off")
plt.title('Test image')
display_predictions(prediction_pairs)
plt.show()
