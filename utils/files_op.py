import os
from pathlib import Path

ROOT_DIR = Path('.')
DB_DIR = os.path.join(ROOT_DIR, 'database')


class Files:
    def __init__(self, db_name, parent_dir=DB_DIR):
        img_dir = os.path.join(parent_dir, db_name)
        print(img_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        self.img_dir = img_dir
        self.files = [os.path.join(self.img_dir, file) for file in os.listdir(self.img_dir)]
        self.files_length = len(self.files)

    def rename_files(self, prefix=""):
        files = self.files
        for i in range(len(files)):
            new_file_name = self.img_dir + '\\' f'{prefix}_{i:04d}.jpg'
            os.rename(files[i], new_file_name)



# def img_files(dir_name='r_m'):
#
#     img_dir = os.path.join(DB_DIR, dir_name)
#     files = [os.path.join(img_dir, file) for file in os.listdir(img_dir)]
#     return files
#
# def rename_file(files, dir):
#     for i in range(len(files)):
#         new_file_name = dir + '//' f'{prefix}_{i:04d}.jpg'
#         print(files[i])
#         print(new_file_name)
#         os.rename(files[i], new_file_name)


# def img_dir(img_dir_name, DB_DIR):
#     img_dir_path = os.path.join(DB_DIR, img_dir_name)
#     if not os.path.exists(img_dir_path):
#         os.makedirs(img_dir_path)




