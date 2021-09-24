import os

SOURCE_DIRECTORY = '/home/stefan/Documents/ObjectDetection/dataset/roboflow_self_driving_small/export'

continuousDuplicates = 4

sorted_files = sorted(os.listdir(SOURCE_DIRECTORY))
for index, file_name in enumerate(sorted_files):
    file_path = os.path.join(SOURCE_DIRECTORY, file_name)
    if index % continuousDuplicates >= 2:
        os.remove(file_path)
