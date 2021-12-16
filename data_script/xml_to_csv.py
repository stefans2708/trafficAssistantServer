import glob
import os.path
import xml.etree.ElementTree as ET

import pandas as pd

IMAGES_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\training'
FILE_NAME = 'training.csv'
DST_DIR = 'C:\\Users\\stefa\\Desktop\\'

xml_list = []
for xml_file in glob.glob(IMAGES_DIRECTORY + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        bbox = member.find('bndbox')
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 member[0].text,
                 member.find('truncated').text,
                 member.find('difficult').text,
                 int(bbox.find('xmin').text),
                 int(bbox.find('ymin').text),
                 int(bbox.find('xmax').text),
                 int(bbox.find('ymax').text)
                 )
        xml_list.append(value)

column_name = ['filename', 'width', 'height', 'class', 'truncated', 'difficult', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv(os.path.join(DST_DIR, FILE_NAME), index=False)
print('Successfully converted xml to csv.')
