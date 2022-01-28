import cv2
import mediapipe as mp
import math
import numpy as np
from skimage.exposure import match_histograms
"""
input = image format -> output = transformed image
Functions:
    resize(): choose width. Puts black padding to fil image.
    nose_mas(): Apply filter to remove eyes on the nose.
"""


# Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.75)


class Image:

    def __init__(self, img):
        self.img = img
        self.shape = img.shape
        self.height, self.width = img.shape[:2]

    def resize(self, img_rename='face', r_width=400):
        rescale = r_width/self.shape[1]
        r_height = int(rescale*self.shape[0])
        dim = (r_width, r_height)
        r_img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)
        self.img = r_img


    def show_img(self):
        cv2.imshow('test', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mesh_pts(self, draw=False, min_detect=0.7):
        # Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=min_detect)
        try:
            # Exception error when landmarks are not found

            result = face_mesh.process(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            print('Result', result)
            for facial_landmarks in result.multi_face_landmarks:
                for i in range(0,468):
                    pt = facial_landmarks.landmark[i]
                    x = int(pt.x * self.width)
                    y = int(pt.y * self.height)
                    pt_nb = str(i)
                    if draw:
                        cv2.circle(self.img, (x, y), 5, (100, 100, 0), -1)
                        cv2.putText(self.img, pt_nb, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (2, 255, 0), 1, cv2.LINE_AA)

            mesh_pts = result.multi_face_landmarks[0]
            if mesh_pts == 0:
                raise ValueError('Face not found')
            return mesh_pts
        except:
            raise ValueError('Face not found (except line)')
            return 0

    def align(self, mesh_pts):
        x263 = mesh_pts.landmark[263].x
        x33 = mesh_pts.landmark[33].x
        y33 = mesh_pts.landmark[33].y
        y263 = mesh_pts.landmark[263].y
        # wrapAffine takes the angle in deg
        angle_rad = math.atan((y263-y33)/(x263-x33))
        angle = angle_rad*(360/(2*math.pi))
        print('Rotation angle: ', angle)

        # Rotation to do on axe center not image center
        rot = cv2.getRotationMatrix2D((self.width/2, self.height/2), angle, 1)

        # rot = cv2.getRotationMatrix2D((int(x263-x33)/2, int(y263-y33)/2), angle, 1)
        al_img = cv2.warpAffine(self.img, rot, (self.width, self.height))
        self.img = al_img
        return al_img

    # Only works when reducing sizes
    def resize_img(self, size=(50, 50)):

        h, w = self.shape[:2]
        c = self.shape[2] if len(self.shape) > 2 else 1

        if h == w:
            return cv2.resize(self.img, size, cv2.INTER_AREA)

        dif = h if h > w else w

        interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

        x_pos = (dif - w) // 2
        y_pos = (dif - h) // 2

        if len(self.shape) == 2:
            mask = np.zeros((dif, dif), dtype=self.img.dtype)
            mask[y_pos:y_pos + h, x_pos:x_pos + w] = self.img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=self.img.dtype)
            mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = self.img[:h, :w, :]

        self.img = cv2.resize(mask, size, interpolation)
        self.width, self.height = size[:2]

        self.shape = size + (c,)


    def nose_mask(self):
        c1 = (0, 0)
        c2 = (self.width, 0)
        radius = int(self.width // 3)
        color = (0, 0, 0)
        thickness = -1
        self.img = cv2.circle(self.img, c1, radius, color, thickness)
        self.img = cv2.circle(self.img, c2, radius, color, thickness)

    def gray_color(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)


    def normalize(self, rgb_min, rgb_max):
        norm_img = np.zeros((300, 300))
        self.img = cv2.normalize(self.img, norm_img, rgb_min, rgb_max, cv2.NORM_MINMAX)

    def matching_hist(self, facial):
        reference = cv2.imread(f'utils/ref_img/{facial}.jpg')
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        # if not gray...
        self.img = match_histograms(self.img, reference).astype(np.uint8)

    def matching_hist_color(self, facial):
        reference = cv2.imread(f'utils/ref_img/{facial}.jpg')
        # reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        # if not gray...
        self.img = match_histograms(self.img, reference).astype(np.uint8)

    def g_bluring(self, blur=(5,5)):
        self.img = cv2.GaussianBlur(self.img, blur, cv2.BORDER_DEFAULT)

    def edge_detection(self, min=10, max=40):
        self.img = cv2.Canny(self.img, min, max)

    # Change contrast and luminosity
    def alpha_beta_correction(self, alpha=1.3, beta=30):
        self.img = cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)


    def alpha_correction(self, alpha, beta=0):
        self.img = cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)

    def horizontal_line(self):
        edges = cv2.Canny(self.img, 80, 120)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 2, None, 30, 1);
        for line in lines[0]:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(self.img, pt1, pt2, (0, 0, 255), 3)


    def binary_conversion(self):
        # Thresholding the image
        (thresh, img_bin) = cv2.threshold(self.img, 158, 255, cv2.THRESH_BINARY)
        # Invert the image
        self.img =  img_bin

##### CROPPING FUNCTIONS #########
    def center_crop(self, cord, t_size):

        t_height, t_width = t_size
        r_la = t_height / t_width
        size = (t_width, t_height)

        x0, y0, x1, y1 = cord
        print(x0, y0, x1, y1)
        pad = int((y1 - y0) / r_la - (x1 - x0))

        pad2 = pad // 2
        if pad < 0:

            # add to the bottom
            r_crop_img = self.img[y0+pad2:y1-pad2, x0:x1]
            print('Pad added top and bot')


        elif pad >= 0:
            # add to both side
            pad2 = pad // 2
            r_crop_img = self.img[y0:y1, x0 - pad2:x1 + pad2]
            print('Pad added both sides left and right, size cropped: ', Image(r_crop_img).shape)


        r_crop_img = cv2.resize(r_crop_img, size, interpolation=cv2.INTER_AREA)

        # cv2.imwrite(FACIAL_DIR + f'//{facial}_{i:04d}.jpg', r_crop_img)
        # print('ratio: ', r_crop_img.shape[0] / r_crop_img.shape[1])
        print('PAD : ', pad)
        # print('Target ratio : ', r_la)
        self.img = r_crop_img
        return r_crop_img


    def contours(self, pts_crop, db, line):
        col_nb = db.columns.levels[0]
        abs_pts = []
        for pt_crop in pts_crop:
            abs_pt = [int((db.loc[line, (col_nb[pt_crop], 'x')]) * self.width), int(db.loc[line, (col_nb[pt_crop], 'y')] * self.height)]
            abs_pts.append(abs_pt)
        # print(abs_pts)
        # abs_pts = np.array(abs_pts).astype(np.uint8)
        # print(abs_pts)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, np.int32([abs_pts]), (255))
        # Image(mask).show_img()

        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 10, cv2.THRESH_BINARY)

        res[thresh == 0] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(res, kernel, iterations=1)
        return erosion

    def contours_in_out(self, pts_outer, pts_inner, db, line):
        col_nb = db.columns.levels[0]
        outer_pts = []
        for pt_crop in pts_outer:
            abs_pt = [int((db.loc[line, (col_nb[pt_crop], 'x')]) * self.width), int(db.loc[line, (col_nb[pt_crop], 'y')] * self.height)]
            outer_pts.append(abs_pt)

        inner_pts = []
        for pt_crop in pts_inner:
            abs_pt = [int((db.loc[line, (col_nb[pt_crop], 'x')]) * self.width), int(db.loc[line, (col_nb[pt_crop], 'y')] * self.height)]
            inner_pts.append(abs_pt)

        # print(self.height, self.width)
        # abs_pts = np.array(abs_pts).astype(np.uint8)
        # print(abs_pts)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, np.int32([outer_pts]), (255))
        cv2.fillPoly(mask, np.int32([inner_pts]), (0))



        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 10, cv2.THRESH_BINARY)

        res[thresh == 0] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(res, kernel, iterations=1)

        return erosion

    def blur_face(self, pts_face, db, col_nb, line):
        blur_pts = []
        blur = (5,5)
        for pt_crop in pts_face:
            abs_pt = [int((db.loc[line, (col_nb[pt_crop], 'x')]) * self.width), int(db.loc[line, (col_nb[pt_crop], 'y')] * self.height)]
            blur_pts.append(abs_pt)


        cv2.GaussianBlur(self.img, blur, cv2.BORDER_DEFAULT)

        # print(self.height, self.width)
        # abs_pts = np.array(abs_pts).astype(np.uint8)
        # print(abs_pts)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)



        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 10, cv2.THRESH_BINARY)

        res[thresh == 0] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(res, kernel, iterations=1)

        return erosion


    def contours_bitwise(self, pts_crop, db, line):
        col_nb = db.columns.levels[0]
        abs_pts = []
        for pt_crop in pts_crop:
            abs_pt = [int((db.loc[line, (col_nb[pt_crop], 'x')]) * self.width), int(db.loc[line, (col_nb[pt_crop], 'y')] * self.height)]
            abs_pts.append(abs_pt)
        # print(abs_pts)
        # abs_pts = np.array(abs_pts).astype(np.uint8)
        # print(abs_pts)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, np.int32([abs_pts]), (255))


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(mask, kernel, iterations=1)
        return erosion






