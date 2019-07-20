# coding: UTF-8

import numpy as np
from PIL import Image
import cv2

def data_augmentation(batch, resize=(512,512)):
    # Hyper Params
    CROP_X_LEFT = 143
    CROP_Y_LEFT = 302
    CROP_X_RIGHT = 1530
    CROP_Y_RIGHT = 984
    BACKGROUND_COLOR = 22
    # Data Augmentation: <batch: numpy.array>
    new_batch = []
    rands = np.random.rand(batch.shape[0], 5)
    randns = np.random.randn(batch.shape[0], 5)
    for i in range(len(batch)):
        angle = 0.0
        widen = 0
        move_x = 0
        move_y = 0
        gamma = 1.0

        if rands[i][0] > 0.5:
            angle = float(3.0 * randns[i][0])
        if rands[i][1] > 0.5:
            widen = int(50 * randns[i][1])
        if rands[i][2] > 0.5:
            move_x = int(5 * randns[i][2])
        if rands[i][3] > 0.5:
            move_y = int(20 * randns[i][3])
        if rands[i][4] > 0.5:
            gamma = float(1.0 + randns[i][4])
            gamma = max(0.7, gamma)
            gamma = min(gamma, 1.7)

        # grayスケール用
        img = Image.fromarray(batch[i][0])

        # crop
        # img = img.crop((CROP_X_LEFT, CROP_Y_LEFT, CROP_X_RIGHT, CROP_Y_RIGHT))
        # rotate
        img = np.asarray(img)
        img_height, img_width = img.shape[:2]
        center = (int(img_width/2), int(img_height/2))
        r_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rad = np.deg2rad(angle)
        new_width = int(abs(np.sin(rad)*img_height) + abs(np.cos(rad)*img_width))
        new_height = int(abs(np.sin(rad)*img_width) + abs(np.cos(rad)*img_height))
        r_matrix[0, 2] += int((new_width-img_width)/2)
        r_matrix[1, 2] += int((new_height-img_height)/2)
        img_r = cv2.warpAffine(img, r_matrix, (new_width, new_height), borderValue=BACKGROUND_COLOR)
        # resize
        img_r = Image.fromarray(img_r)
        img_r_width, img_r_height = img_r.size
        mat_width = mat_height = img_r_width + widen
        mat = Image.new(img_r.mode, (mat_width, mat_height), BACKGROUND_COLOR)
        mat.paste(img_r, ((mat_width-img_r_width)//2, (mat_height-img_r_height)//2))
        mat_resize = mat.resize(resize, Image.LANCZOS)
        # translate
        t_matrix = np.array([[1,0,move_x], [0,1,move_y]], dtype=np.float32)
        mat_resize = np.asarray(mat_resize)
        mat_t = cv2.warpAffine(mat_resize, t_matrix, resize, borderValue=BACKGROUND_COLOR)
        # gamma
        gamma_cvt = np.zeros((256, 1), dtype='uint8')
        for j in range(256):
            gamma_cvt[j][0] = 255 * (float(j)/255) ** (1.0/gamma)
        img_gamma = cv2.LUT(mat_t, gamma_cvt)

        # save
        # cv2.imwrite("img_gamma_{}.png".format(i), img_gamma)

        # return
        img_gamma = np.asarray(img_gamma, dtype=np.float32)
        img_gamma = np.expand_dims(img_gamma, axis=0)
        new_batch.append(img_gamma)

    return np.asarray(new_batch)

if __name__ == '__main__':
    # check
    from glob import glob
    import os
    root_path = glob('dataset\\train\\good\\*')
    batch = []
    for path in root_path:
        img = Image.open(path)
        img = img.convert('L')
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        batch.append(img)
    batch = np.asarray(batch)
    print('# batch: {}'.format(batch.shape))
    imgs = data_augmentation(batch)
    print('# imgs: {}'.format(imgs.shape))