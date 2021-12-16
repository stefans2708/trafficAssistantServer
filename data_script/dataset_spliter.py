import os
import shutil

IMAGES_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\dataset\\vlad_training'
DESTINATION_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\dataset\\vlad_validation'
VALIDATION_SPLIT = 0.3

src = IMAGES_DIRECTORY
num_of_images = len(os.listdir(src)) // 2
num_of_val_images = num_of_images * VALIDATION_SPLIT
index_increment = num_of_images / num_of_val_images  # same as 1/VALIDATION_SPLIT

result_img_indices = []
index = 0
index_float = 0

while index_float <= num_of_images:
    result_img_indices.append(index)
    index_float += index_increment
    index = round(index_float)

all_files = os.listdir(IMAGES_DIRECTORY)
print(f'all_files: {len(all_files)}, result_img_indices: {len(result_img_indices)}')
for idx in result_img_indices:
    real_index = idx * 2
    img_file = all_files[real_index]
    xml_file = all_files[real_index + 1]
    print(f'{img_file}, {xml_file}')
    shutil.move(os.path.join(IMAGES_DIRECTORY, xml_file), os.path.join(DESTINATION_DIRECTORY, xml_file))
    shutil.move(os.path.join(IMAGES_DIRECTORY, img_file), os.path.join(DESTINATION_DIRECTORY, img_file))
