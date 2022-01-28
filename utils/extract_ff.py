import cv2
import pandas as pd
from utils.img_transformation import Image
import os
from pathlib import Path
import numpy as np

# PATHS
ROOT_DIR = Path('.')
DB_DIR = os.path.join(ROOT_DIR, 'database')
CSV_DIR = os.path.join(DB_DIR, 'csv_dir')
CROPPED_IMG = os.path.join(DB_DIR, 'cropped_img')
# Load csv df
CSV_FILE = os.path.join(CSV_DIR, 'CHICAGO_MIX.csv')
# CSV_FILE = os.path.join(CSV_DIR, 'CELEBA_2K.csv')

df = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0)


class Facial:

    def __init__(self, file, img_nb):

        img_org = cv2.imread(file)
        img_obj = Image(img_org)
        col_nb = df.columns.levels[0]

        ## To delete !!!!!
        # img_obj.resize_img(size=(2444, 1718))
        self.width = img_obj.width
        print(self.width)
        self.height = img_obj.height
        self.col_nb = col_nb
        self.img_nb = img_nb
        self.df = df

        # img_obj.show_img()
        self.img = img_obj.img


    def get_coordinates(self, coord, size):
        x0 = int((self.df.loc[self.img_nb, (self.col_nb[coord[0]], 'x')] * self.width)-0.1*size[1])
        y0 = int((self.df.loc[self.img_nb, (self.col_nb[coord[1]], 'y')] * self.height)-0.1*size[0])
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[coord[2]], 'x')] * self.width)+0.1*size[1])
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[coord[3]], 'y')]) * self.height+0.1*size[0])
        coord = x0, y0, x1, y1
        return coord


    def lips_ex2(self):
        # add black at top right corner
        cord = [57, 164, 287, 18]
        size = (200, 400) # height, width
        facial = 'lips'

        # col_nb = db.columns.levels[0]
        # Crop polygons
        pts_crop = np.array([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                             375, 321, 405, 314, 17, 84, 181, 91, 146])

        # res = Image(self.img).contours(pts_crop, self.df, self.img_nb)
        x0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[0]], 'x')] * self.width)-0.1*size[1])
        y0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[1]], 'y')] * self.height)-0.1*size[0])
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[2]], 'x')] * self.width)+0.1*size[1])
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[3]], 'y')]) * self.height+0.1*size[0])
        cord_abs = x0, y0, x1, y1
        facial_img = Image(self.img).center_crop(cord_abs, size)

        return facial_img


    def nose_ex3(self):
        nose_cord = [129, 197, 358, 2]
        size = (200, 256)
        facial = 'nose3'
        pts_crop = np.array([6, 351, 412, 399, 456, 363, 360, 360, 279, 358,327, 326,
                             2, 97, 98, 129, 49, 131, 134, 236, 174, 188, 122, 6])

        # res = Image(self.img).contours(pts_crop, self.df, self.img_nb)
        x0 = int((self.df.loc[self.img_nb, (self.col_nb[129], 'x')] * self.width)-0.1*size[1])
        y0 = int((self.df.loc[self.img_nb, (self.col_nb[197], 'y')] * self.height)-0.1*size[0])
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[358], 'x')] * self.width)+0.1*size[1])
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[2], 'y')]) * self.height+0.1*size[0])
        cord = x0, y0, x1, y1
        # print('image shape', Image(self.img).shape)
        facial_img = Image(self.img).center_crop(cord, size)
        # print('image shape', Image(self.img).shape)
        return facial_img

    def l_eyes_ex(self):
        nose_cord = [463, 257, 359, 253]
        size = (256, 512)
        facial = 'l_eyes_v2'



        # res = Image(self.img).contours_in_out(outer_crop, inner_crop, self.df, self.img_nb)
        # Image(res).show_img()
        coord = self.get_coordinates(nose_cord, size)
        # print(coord)
        facial_img = Image(self.img).center_crop(coord, size)

        return facial_img

    def eyebrow(self):
        # add black at top right corner
        cord = [285, 298, 300, 444]
        size = (200, 512)
        facial = 'eyebrow'

        # col_nb = db.columns.levels[0]
        # Crop polygons
        pts_crop = np.array([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                             375, 321, 405, 314, 17, 84, 181, 91, 146])

        # res = Image(self.img).contours(pts_crop, self.df, self.img_nb)
        # x0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[0]], 'x')] * self.width)-0.1*size[1])
        # y0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[1]], 'y')] * self.height)-0.1*size[0])
        # x1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[2]], 'x')] * self.width)+0.1*size[1])
        # y1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[3]], 'y')]) * self.height+0.1*size[0])
        x0 = int(self.df.loc[self.img_nb, (self.col_nb[cord[0]], 'x')] * self.width)
        y0 = int(self.df.loc[self.img_nb, (self.col_nb[cord[1]], 'y')] * self.height)
        x1 = int(self.df.loc[self.img_nb, (self.col_nb[cord[2]], 'x')] * self.width)
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[3]], 'y')]) * self.height)
        cord_abs = x0, y0, x1, y1
        facial_img = Image(self.img).center_crop(cord_abs, size)

        return facial_img

    def faceshape(self):
        # add black at top right corner
        cord = [227, 10, 447, 175]
        size = (512, 512)
        facial = 'faceshape'

        # col_nb = db.columns.levels[0]
        # Crop polygons
        pts_crop = np.array([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                            288, 397, 365, 379, 400, 377, 152, 148, 176, 150, 136, 172,
                             58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10])

        res = Image(self.img).contours_bitwise(pts_crop, self.df, self.img_nb)
        x0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[0]], 'x')] * self.width)-0.1*size[1])
        y0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[1]], 'y')] * self.height)-0.1*size[0])
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[2]], 'x')] * self.width)+0.1*size[1])
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[3]], 'y')]) * self.height+0.1*size[0])
        cord_abs = x0, y0, x1, y1
        facial_img = Image(res).center_crop(cord_abs, size)

        return facial_img

    def jaw(self):
        # add black at top right corner
        cord = [227, 10, 447, 175]
        cord = [227, 234, 447, 175]
        size = (512, 512)

        # col_nb = db.columns.levels[0]
        # Crop polygons
        pts_crop = np.array([454, 323, 361,
                            288, 397, 365, 379, 400, 377, 152, 148, 176, 150, 136, 172,
                             58, 132, 93, 234, 454])

        res = Image(self.img).contours_bitwise(pts_crop, self.df, self.img_nb)
        x0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[0]], 'x')] * self.width)-0.2*size[1])
        y0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[1]], 'y')] * self.height)-0.2*size[0])
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[2]], 'x')] * self.width)+0.2*size[1])
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[3]], 'y')]) * self.height+0.2*size[0])
        cord_abs = x0, y0, x1, y1
        facial_img = Image(res).center_crop(cord_abs, size)

        return facial_img

    def ears(self):
        # add black at top right corner
        cord = [227, 10, 447, 175]
        cord = [234]
        size = (512, 128)

        # col_nb = db.columns.levels[0]
        # Crop polygons
        pts_crop = np.array([454, 323, 361,
                            288, 397, 365, 379, 400, 377, 152, 148, 176, 150, 136, 172,
                             58, 132, 93, 234, 454])

        h_width = 0.2*(self.df.loc[self.img_nb, (self.col_nb[454], 'x')] - self.df.loc[self.img_nb, (self.col_nb[234], 'x')])

        # res = Image(self.img).contours(pts_crop, self.df, self.img_nb)
        x0 = int(((self.df.loc[self.img_nb, (self.col_nb[234], 'x')]-h_width) * self.width))
        y0 = int((self.df.loc[self.img_nb, (self.col_nb[162], 'y')] * self.height))
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[234], 'x')] * self.width)+0.2*size[1])
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[132], 'y')]) * self.height)
        cord_abs = x0, y0, x1, y1
        facial_img = Image(self.img).center_crop(cord_abs, size)

        return facial_img



    def facialhair(self):
        # add black at top right corner
        cord = [227, 234, 447, 175]
        size = (128, 256)

        x0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[0]], 'x')] * self.width)-0.1*size[1])
        y0 = int((self.df.loc[self.img_nb, (self.col_nb[cord[1]], 'y')] * self.height)-0.1*size[0])
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[2]], 'x')] * self.width)+0.1*size[1])
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[cord[3]], 'y')]) * self.height+0.1*size[0])
        cord_abs = x0, y0, x1, y1
        img_obj = Image(self.img)

        img_obj.center_crop(cord_abs, size)
        img_obj.g_bluring(blur=(9, 9))
        return img_obj.img



    def hair(self):

        # add black at top right corner
        pts_crop= [227, 10, 447, 175]
        pts_crop= [227, 151, 447, 18]
        size = (1024, 1024)
        facial = 'lips'
        abs_pts = []
        for pt_crop in pts_crop:
            abs_pt = [int((df.loc[self.img_nb, (self.col_nb[pt_crop], 'x')]) * self.width), int(self.df.loc[self.img_nb, (self.col_nb[pt_crop], 'y')] * self.height)]
            abs_pts.append(abs_pt)
        # facial_img = Image(res).center_crop(cord, size)
        mask = np.ones((self.height, self.width), dtype=np.uint8)*255
        cv2.fillPoly(mask, np.int32([abs_pts]), (0))
        # mask = np.stack((mask,)*3, axis=-1).astype('uint32')
        # print('mask dimension ', mask.shape)

        # Image(mask).show_img()
        blur_copy_obj = Image(self.img)
        # blur_copy_obj.g_bluring(blur=(2221,2221))
        # print('blur dimension  ', blur_copy_obj.shape)
        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        res_obj = Image(res)
        res_obj.resize_img(size=(1024, 1024))
        # Image(res).show_img()
        return res_obj.img


    def hairtop(self):
        # add black at top right corner
        cord = [68, 298]
        size = (128, 256)
        h_width = 0.5*(self.df.loc[self.img_nb, (self.col_nb[284], 'x')] - self.df.loc[self.img_nb, (self.col_nb[54], 'x')])
        print('h_width: ', h_width)
        # res = Image(self.img).contours(pts_crop, self.df, self.img_nb)

        x0 = int(((self.df.loc[self.img_nb, (self.col_nb[54], 'x')]) * self.width))
        y0 = int(((self.df.loc[self.img_nb, (self.col_nb[54], 'y')] - h_width) * self.height))
        x1 = int((self.df.loc[self.img_nb, (self.col_nb[284], 'x')] * self.width))
        y1 = int((self.df.loc[self.img_nb, (self.col_nb[54], 'y')]) * self.height)
        cord_abs = x0, y0, x1, y1
        facial_img = Image(self.img).center_crop(cord_abs, size)

        return facial_img






    # def forehead_ex(self):
    #     cord = [69, 67, 299, 333]
    #     size = (200, 900)
    #     facial = 'forehead'
    #     # Image(res).show_img()
    #     coord = self.get_coordinates(cord, size)
    #     # print(coord)
    #     facial_img = Image(self.img).center_crop(coord, size)
    #     height, width = facial_img.shape[:2]
    #     start_point = (0, 0)
    #     end_point = (int(width), int(0.3*height))
    #     side1e = (int(0.1*width), int(height))
    #     side2s = (int(0.9*width), int(0*height))
    #     side2e = (int(width), int(height))
    #     color = (200, 200, 200)
    #     thickness = -1
    #     facial_img = cv2.rectangle(facial_img, start_point, end_point, color, thickness)
    #     facial_img = cv2.rectangle(facial_img, start_point, side1e, color, thickness)
    #     facial_img = cv2.rectangle(facial_img, side2s, side2e, color, thickness)
    #     # Image(facial_img).show_img()
    #     return facial_img
    #
    #
    #
    #
    #
    #
    #












    # def eye_line_ex(self):
    #     cord = [465, 341, 347, 347]
    #     size = (45,100)
    #     facial = 'l_eye_line'
    #     # Image(res).show_img()
    #     coord = self.get_coordinates(cord, size)
    #     # print(coord)
    #     facial_img = Image(self.img).center_crop(coord, size)
    #
    #     height, width = facial_img.shape[:2]
    #     start_point = (int(0.3*width), 0)
    #     end_point = (int(width), int(0.3*height))
    #     color = (200, 200, 200)
    #     thickness = -1
    #     facial_img = cv2.rectangle(facial_img, start_point, end_point, color, thickness)
    #     # Image(facial_img).show_img()
    #     return facial_img
    #

    #
    # def lips_ex(self):
    #     # add black at top right corner
    #     cord = [61, 164, 291, 18]
    #     size = (40, 80)
    #     facial = 'lips'
    #
    #     # col_nb = db.columns.levels[0]
    #     # Crop polygons
    #     pts_crop = np.array([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    #                          375, 321, 405, 314, 17, 84, 181, 91, 146])
    #
    #     res = Image(self.img).contours(pts_crop, self.df, self.img_nb)
    #     x0 = int((self.df.loc[self.img_nb, (self.col_nb[61], 'x')] * self.width)-0.1*size[1])
    #     y0 = int((self.df.loc[self.img_nb, (self.col_nb[0], 'y')] * self.height)-0.1*size[0])
    #     x1 = int((self.df.loc[self.img_nb, (self.col_nb[291], 'x')] * self.width)+0.1*size[1])
    #     y1 = int((self.df.loc[self.img_nb, (self.col_nb[17], 'y')]) * self.height+0.1*size[0])
    #     cord = x0, y0, x1, y1
    #     facial_img = Image(res).center_crop(cord, size)
    #
    #     return facial_img


    # def nose_ex(self):
    #     nose_cord = [129, 197, 358, 2]
    #     size = (80, 100)
    #     facial = 'nose'
    #     pts_crop = np.array([6, 351, 412, 399, 456, 363, 360, 360, 279, 358,327, 326,
    #                          2, 97, 98, 129, 49, 131, 134, 236, 174, 188, 122, 6])
    #
    #     res = Image(self.img).contours(pts_crop, self.df, self.img_nb)
    #     x0 = int((self.df.loc[self.img_nb, (self.col_nb[129], 'x')] * self.width)-0.1*size[1])
    #     y0 = int((self.df.loc[self.img_nb, (self.col_nb[6], 'y')] * self.height)-0.1*size[0])
    #     x1 = int((self.df.loc[self.img_nb, (self.col_nb[358], 'x')] * self.width)+0.1*size[1])
    #     y1 = int((self.df.loc[self.img_nb, (self.col_nb[2], 'y')]) * self.height+0.1*size[0])
    #     cord = x0, y0, x1, y1
    #     facial_img = Image(self.img).center_crop(cord, size)
    #
    #     return facial_img

