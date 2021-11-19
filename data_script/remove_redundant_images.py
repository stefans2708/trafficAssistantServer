import os

SOURCE_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\test'

continuousDuplicates = 4

sorted_files = sorted(os.listdir(SOURCE_DIRECTORY))
for index, file_name in enumerate(sorted_files):
    file_path = os.path.join(SOURCE_DIRECTORY, file_name)
    if index % continuousDuplicates >= 2:
        os.remove(file_path)
