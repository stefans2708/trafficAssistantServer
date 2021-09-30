import glob
import os.path
import xml.etree.ElementTree as ET

import pandas as pd

IMAGES_DIRECTORY = '/home/stefan/Documents/ObjectDetection/TrafficAssistant/validation/'
FILE_NAME = 'validation_labels.csv'
DST_DIR = '/home/stefan/Documents/ObjectDetection/TrafficAssistant/'

xml_list = []
for xml_file in glob.glob(IMAGES_DIRECTORY + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 member[0].text,
                 int(member[5][0].text),
                 int(member[5][2].text),
                 int(member[5][1].text),
                 int(member[5][3].text)
                 )
        xml_list.append(value)

column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv(os.path.join(DST_DIR, FILE_NAME), index=False)
print('Successfully converted xml to csv.')
