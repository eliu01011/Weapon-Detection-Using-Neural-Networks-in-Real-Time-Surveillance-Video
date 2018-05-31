"""
:Credit https://stackoverflow.com/questions/23556040/moving-specific-file-types-with-python
Taken from link above and modified
"""

import os
import shutil
import random
import os.path

src_dir = 'C:/Users/vanes/Pictures/images/My_image/SELECTED_GUNS'
# ALL GUNS/ ALL KNIVES/ SELECTED_GUNS/ SELECTED_KNIVES
target_dir = 'C:/Users/vanes/Pictures/images/My_image/TEST'
# SELECTED_KNIVES/ SELECTED_GUNS/ TRAIN/ TEST
src_files = (os.listdir(src_dir))


def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)


files = [os.path.join(src_dir, f) for f in src_files if valid_path(src_dir, f) and f.endswith('.jpg')]
choices = random.sample(files, 250)

mylist = []
for files in choices:
    shutil.move(files, target_dir)
    # shutil.copy for select random images to SELECTED folders
    # shutil.move for moving random 25% to TEST
    mylist.append(str(files).replace(".jpg", ".xml"))


for xml in mylist:
    # xml = jpg.replace(".jpg", ".xml")
    shutil.move(xml, target_dir)
    # shutil.copy for select random images to SELECTED folders
    # shutil.move for moving random 25% to TEST
    # print(xml)

print(len(mylist))
print('Finished!')
