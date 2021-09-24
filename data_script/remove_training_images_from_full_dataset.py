import os
import shutil

TRAINING_SET_DIRECTORY = '/home/stefan/Documents/ObjectDetection/dataset/roboflow_self_driving_small_training/export'
FULL_DATASET_DIRECTORY = '/home/stefan/Documents/ObjectDetection/dataset/roboflow_self_driving_small/export'
VALIDATION_SET_DIRECTORY = '/home/stefan/Documents/ObjectDetection/dataset/roboflow_self_driving_small_validation'

training_files = sorted(os.listdir(TRAINING_SET_DIRECTORY))
training_data_map = dict()

for i in range(0, len(training_files), 2):
    if training_files[i][:-3] != training_files[i + 1][:-3]:
        print(f'File {training_files[i]} not equal to {training_files[i + 1]}')
        raise ValueError

files_count = len(training_files)
for i in range(0, files_count, 2):
    training_data_map[training_files[i]] = training_files[i + 1]

files = sorted(os.listdir(FULL_DATASET_DIRECTORY))
files_count = len(files)
for i in range(0, files_count, 2):
    if not (files[i] in training_data_map):
        shutil.copy(os.path.join(FULL_DATASET_DIRECTORY, files[i]), VALIDATION_SET_DIRECTORY)
        shutil.copy(os.path.join(FULL_DATASET_DIRECTORY, files[i + 1]), VALIDATION_SET_DIRECTORY)
