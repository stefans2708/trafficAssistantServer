import os
import xml.etree.ElementTree as ET

SOURCE_DIRECTORY = '/home/stefan/Documents/ObjectDetection/dataset/roboflow_self_driving_small/export'

for file_name in os.listdir(SOURCE_DIRECTORY):
    file_path = os.path.join(SOURCE_DIRECTORY, file_name)
    ext = file_name[-4:]
    image_name = file_name[:-4]+'.jpg'
    if ext == '.xml':
        tree = ET.parse(file_path)
        root = tree.getroot()
        root[1].text = image_name
        root[2].text = image_name
        tree.write(file_path)
