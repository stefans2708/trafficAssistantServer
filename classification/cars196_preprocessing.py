import os.path

from cv2 import cv2
from scipy.io import loadmat

PATH = 'C:\\Users\\stefa\\Desktop\\cars196'
PATH_RESULT = 'C:\\Users\\stefa\\Desktop\\cars196\\result'
mat_all = loadmat(os.path.join(PATH, 'devkit\\cars_annos.mat'))
meta = loadmat(os.path.join(PATH, 'devkit\\cars_meta.mat'))

labels = list()
for lbl in meta['class_names'][0]:
    labels.append(lbl[0])

# ('car_ims/000001.jpg', 112, 7, 853, 717, 1)
for example in mat_all['annotations'][0]:
    image = example[0][0]
    bbox_x1 = example[1][0][0]
    bbox_y1 = example[2][0][0]
    bbox_x2 = example[3][0][0]
    bbox_y2 = example[4][0][0]
    class_car = example[5][0][0]

    image_file_name = image.split('/')[1]
    car_model = labels[class_car - 1].replace('/', '_')
    car_model_folder = os.path.join(PATH_RESULT, car_model)
    if not os.path.exists(car_model_folder):
        os.mkdir(car_model_folder)

    img = cv2.imread(os.path.join(PATH, image))
    img = img[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
    cv2.imwrite(os.path.join(car_model_folder, image_file_name), img)
