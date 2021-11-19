import os
import xml.etree.ElementTree as ET

SOURCE_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\roboflow_dataset'
RELEVANT_CLASSES = ['car', 'truck', 'biker', 'pedestrian', 'bus']

for file_name in os.listdir(SOURCE_DIRECTORY):
    if file_name.endswith('xml'):
        file_path = os.path.join(SOURCE_DIRECTORY, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # change the name of image in proper tag
        # image_name = file_name[:-3] + 'jpg'
        # root[1].text = image_name
        # root[2].text = image_name

        # remove all non-relevant object annotations
        # for relevant change order (xmin,xmax,ymin,ymax -> xmin,ymin,xmax,ymax)
        for obj in root.findall('object'):
            if obj[0].text not in RELEVANT_CLASSES:
                root.remove(obj)
            # else:
            #     # todo: check if this is needed
            #     box = obj.find('bndbox')
            #     x_max = box.find('xmax')
            #     box.remove(x_max)
            #     box.insert(1, x_max)

        tree.write(file_path)
