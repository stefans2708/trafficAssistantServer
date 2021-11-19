import os
import cv2.cv2 as cv2

SOURCE_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\cars'
DESTINATION_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\cars_resized'
DESIRED_IMAGE_SIZE = (512, 512)

for image_name in os.listdir(SOURCE_DIRECTORY):
    img_path = os.path.join(SOURCE_DIRECTORY, image_name)
    # if os.path.isfile(img_path) and (imghdr.what(img_path) == 'jpg' or imghdr.what(img_path) == 'jpeg'):
    img = cv2.imread(img_path)
    img = cv2.resize(img, DESIRED_IMAGE_SIZE)
    cv2.imwrite(os.path.join(DESTINATION_DIRECTORY, image_name), img)
