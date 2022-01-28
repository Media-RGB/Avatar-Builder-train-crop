"""
Extract from db the images, align them and save them in specified race dir
"""
import glob
import os
from pathlib import Path
import cv2
from utils.img_transformation import Image
from utils.files_op import Files


ROOT_DIR = Path('.')
DB_DIR = os.path.join(ROOT_DIR, 'database')

# variables
P_DIR = 'cfd'
race = 'LFW_AL'


def transform(files, dir_name):
    face_saved = 0
    total_img = len(files)
    new_dir = os.path.join(DB_DIR, dir_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for i in range(0, 2000):
        try:
            img = cv2.imread(files[i])
            img = Image(img)
            mesh_pts = img.mesh_pts()

            mod_img = img.align(mesh_pts)

            new_file_path = new_dir + '\\' + f'{dir_name}_{face_saved:04d}.jpg'
            cv2.imwrite(new_file_path, mod_img)
            face_saved += 1
            print(new_file_path)

        except Exception as e:
            print('Except Error: ', e, f' at i = {i}')

    print(f'{face_saved} aligned faces. \n{total_img} total images.')


# RACE_DIR = os.path.join(os.path.join(DB_DIR, 'cfd'), race)
# RACE_DIR = os.path.join(DB_DIR, 'CELAB_A_3k')
# files = glob.glob('/**/*.jpg', recursive=True)
# print(files)

files_obj = Files('CELAB_A_3k')
files = files_obj.files
transform(files, 'CELEBA_2k')