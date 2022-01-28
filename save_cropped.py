"""
Call crop_ff to extract features and save it.
"""

from utils.extract_ff import Facial
import os
from pathlib import Path
from utils.files_op import Files
from utils.pre_ff import PreFacial
from viz.clust_viz import ClusterGraphs
from utils.img_transformation import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

DBS = ['CELEBA_2K', 'CHICAGO_MIX']
IMGS_DIR = DBS[1]

# PATHS
ROOT_DIR = Path('.')
DB_DIR = os.path.join(ROOT_DIR, 'database')
CSV_DIR = os.path.join(DB_DIR, 'csv_dir')
CROPPED_FF_DIR = os.path.join(DB_DIR, 'cropped_ff')

# Load Files
files_obj = Files(IMGS_DIR)
files = files_obj.files
files_length = files_obj.files_length

# Load csv df
CSV_FILE = os.path.join(CSV_DIR, f'{IMGS_DIR}.csv')
df_org = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0)
data = {}
model_arrs = []
files_length = len(files)
# files_length = 10
facial = 'hairtop'

for i in range(files_length):
    try:
        # Contour the selected facial.
        facial_obj = Facial(files[i], i)

        ## Lips
        # facial_img = facial_obj.lips_ex2()
        # facial_img = facial_obj.nose_ex3()
        # facial_img = facial_obj.l_eyes_ex()
        # facial_img = facial_obj.hair()
        facial_img = facial_obj.hairtop()
        # facial_img = facial_obj.facialhair()
        # facial_img = facial_obj.ears()
        # facial_img = facial_obj.jaw()
        # facial_img = facial_obj.faceshape()
        # facial_img = facial_obj.eyebrow()
        # facial_img = facial_obj.forehead_ex()
        # facial_img = facial_obj.eye_line_ex()
        # print('image shape: ', facial_img.shape)
        # Image(facial_img).show_img()
        FACIAL_DIR = os.path.join(CROPPED_FF_DIR, f'{facial}')
        if not os.path.exists(FACIAL_DIR):
            os.makedirs(FACIAL_DIR)
        img_file = os.path.join(FACIAL_DIR, f'{facial}_ID_{i:04d}.jpg')
        cv2.imwrite(img_file, facial_img)
    except Exception as e:
        print('Database line: ', i)
        print('Error: ', e)
        img0 = np.zeros((512, 512))
        FACIAL_DIR = os.path.join(CROPPED_FF_DIR, f'{facial}')
        if not os.path.exists(FACIAL_DIR):
            os.makedirs(FACIAL_DIR)
        img_file = os.path.join(FACIAL_DIR, f'{facial}_ID_{i:04d}.jpg')
        cv2.imwrite(img0, facial_img)








