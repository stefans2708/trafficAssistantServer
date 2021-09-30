import os
import xml.etree.ElementTree as ET

SOURCE_DIRECTORY = '/home/stefan/Documents/ObjectDetection/TrafficAssistant/validation/'
RELEVANT_CLASSES = ['car', 'truck', 'biker', 'pedestrian']

for file_name in os.listdir(SOURCE_DIRECTORY):
    if file_name.endswith('xml'):
        file_path = os.path.join(SOURCE_DIRECTORY, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # change the name of image in proper tag
        image_name = file_name[:-3] + 'jpg'
        root[1].text = image_name
        root[2].text = image_name

        # remove all non-relevant object annotations
        for obj in root.findall('object'):
            if obj[0].text not in RELEVANT_CLASSES:
                root.remove(obj)

        tree.write(file_path)
