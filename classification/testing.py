import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image

# force CPU execution to reduce init time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PATH = 'C:\\Users\\stefa\\Desktop\\cars196'
IMAGE_SIZE = (192, 192)


def load_labels():
    dataset_path = os.path.join(PATH, 'dataset')
    return os.listdir(dataset_path)


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


labels = load_labels()

testImage = image.load_img(path=get_test_image_path('golf4'), target_size=IMAGE_SIZE,
                           interpolation='bilinear')
input_arr = image.img_to_array(testImage)
input_arr = np.array([input_arr])

model = tf.keras.models.load_model('../model/classification/classification_model_4.h5')
predictions = model.predict(input_arr)

# predictions is list of confidence for each class of 196 cars. Classes are sorted by name (1..196)
print(predictions)
rounded_predictions = np.round(predictions, decimals=4)
prediction_pairs = list(enumerate(rounded_predictions[0]))
prediction_pairs.sort(key=lambda pair: pair[1], reverse=True)

print(prediction_pairs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.imshow(testImage)
plt.axis("off")
plt.title('Test image')
display_predictions(prediction_pairs)
plt.show()
