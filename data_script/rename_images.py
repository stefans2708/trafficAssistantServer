import os

SOURCE_DIRECTORY = 'C:\\Users\\stefa\\Desktop\\roboflow'

sorted_files = sorted(os.listdir(SOURCE_DIRECTORY))
for index, file_name in enumerate(sorted_files):
    file_path = os.path.join(SOURCE_DIRECTORY, file_name)
    ext = '.jpg' if index % 2 == 0 else '.xml'
    name = index / 2 if index % 2 == 0 else (index - 1) / 2
    name = str(name).replace('.', '_')
    os.rename(file_path, f'{SOURCE_DIRECTORY}/{name}{ext}')
