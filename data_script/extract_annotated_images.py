import glob
import os
import shutil

import cv2.cv2 as cv2

IMAGES_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\cars_resized'
DESTINATION_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\cars_annotated'


def _convert_to_jpg_and_move(img_path, _file_name):
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join(DESTINATION_DIRECTORY, _file_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    os.remove(img_path)


# option 2: in source files leave only non-annotated images
for xml_file in glob.glob(IMAGES_DIRECTORY + '/*.xml'):
    file_full_path = xml_file[:-4]
    file_name = os.path.basename(xml_file)[:-4]
    output_file_name = f'{file_name}.jpg'

    ext = 'xml'
    shutil.move(xml_file, os.path.join(DESTINATION_DIRECTORY, f'{file_name}.{ext}'))

    jpg_path = f'{file_full_path}.jpg'
    jpeg_path = f'{file_full_path}.jpeg'
    png_path = f'{file_full_path}.png'
    bmp_path = f'{file_full_path}.bmp'
    if os.path.exists(jpg_path):
        shutil.move(jpg_path, os.path.join(DESTINATION_DIRECTORY, output_file_name))
    elif os.path.exists(jpeg_path):
        _convert_to_jpg_and_move(jpeg_path, _file_name=output_file_name)
    elif os.path.exists(png_path):
        _convert_to_jpg_and_move(png_path, _file_name=output_file_name)
    else:
        _convert_to_jpg_and_move(bmp_path, _file_name=output_file_name)
