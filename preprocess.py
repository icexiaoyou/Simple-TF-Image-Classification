# First of all, put the pictures in directories according classification
# Secondly, back your foulder which would be processed
# The script will rename and resize all pictures in pic_source, store the processed pictures according to classification
# target_size should be same in four script

import os
from PIL import Image


# ----------Custom Inputs for Scripts----------
input_path = 'voc_dataset'           # Path of the source pictures
target_size = (640, 640)            # Target size of new picture
# ----------That's ALL You Need to Do----------


# Return the number and names of subfolders(classes)
def count_class(folder_path):
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    num = len(subfolders)
    return num, subfolders

# Generate the list of subfolder paths
num_classes, list_classes = count_class(input_path)
list_path_subfolder = []
print('The number of classes is %d' % (num_classes))
print('The paths of subfolders', end=':   ')
for classes in list_classes:
    path_subfolder = os.path.join(input_path, classes)
    list_path_subfolder.append(path_subfolder)
    print(path_subfolder, end='   ')

# Get all pictures in a folder, including .jpg/.png/.jpeg
def get_pics(folder_path):
    pics = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
            pics.append(os.path.join(folder_path, file_name))
    return pics

# Rename all pictures of a foulder, picture new name is "foulder_num"
# At the same time, resize picture to the specified pixel size
def rename_resize_pics(folder_path):
    # Get folder name
    folder_name = os.path.basename(folder_path)
    # Get all pictures
    pics = get_pics(folder_path)
    # rename and resize each picture
    for i, image_path in enumerate(pics):
        # Resize
        img = Image.open(image_path)
        img = img.resize(target_size)
        img.save(image_path)
        # Rename
        image_extension = os.path.splitext(image_path)[1]
        new_image_name = folder_name + '_' + str(i+1) + image_extension    # define connection symbol
        new_image_path = os.path.join(folder_path, new_image_name)
        os.rename(image_path, new_image_path)

# Rename all pictures of all subfolders
for path_subfolder in list_path_subfolder:
    rename_resize_pics(path_subfolder)

print()
print('Preprocess has done!')