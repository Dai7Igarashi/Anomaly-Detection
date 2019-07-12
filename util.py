# coding: UTF-8

from glob import glob
import os
from PIL import Image
import argparse
import numpy as np
import pickle

"""
# in_path  = dataset\\row\\metal_nut\\train\\good\\000.png
# out_path = dataset\\resize_128_128\\metal_nut\\train\\good
"""
def resize_img(in_path, out_path, size=(128,128), mode='LANCZOS'):
    img_name = in_path.split('\\')[-1]
    img = Image.open(in_path)
    if mode == 'LANCZOS':
        _mode = Image.LANCZOS
    elif mode == 'BICUBIC':
        _mode = Image.BICUBIC
    else:
        _mode = Image.NEAREST
    img_resize = img.resize(size, _mode)
    img_resize.save(os.path.join(out_path, img_name))

"""
# wrapper_path  = dataset\\resize_128_128\\metal_nut\\train\\good
# out_path      = dataset\\resize_128_128\\metal_nut
# save_name: ex. train_good
"""
def transform_to_numpy(wrapper_path, out_path, save_name, isMiniMax=False):
    img_list = []

    for in_path in glob(os.path.join(wrapper_path, '*')):
        img = Image.open(in_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = np.asarray(img)
        if isMiniMax:
            img = img / 255.
        img_list.append(img)

    # np arrayとして保存
    img_list = np.asarray(img_list)
    if len(img_list.shape) == 3:
        img_list = np.expand_dims(img_list, axis=3)
    img_list = img_list.transpose(0, 3, 1, 2)
    with open(os.path.join(out_path, '{}.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(img_list, f)

"""
画像リサイズ
"""
def resize_main(args):
    wh = args.size.split(',')
    size = (int(wh[0]), int(wh[1]))
    resize_path_name = 'resize_{}_{}_{}'.format(size[0], size[1], args.mode)
    # resize_path = 'dataset/resize_128_128'
    resize_path = os.path.join('dataset', resize_path_name)
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)

    # material_list = ['dataset/row\\metal_nut']
    material_list = glob('dataset\\row\\*')
    for material in material_list:
        # material_name: ex. 'metal_nut'
        material_name = material.split('\\')[-1]
        # section_list = ['dataset/row\\metal_nut\\ground_truth\\', 'dataset/row\\metal_nut\\test\\', 'dataset/row\\metal_nut\\train\\']
        section_list = glob(os.path.join(material, '*\\'))

        for section in section_list:
            # section_name: ex. 'train'
            section_name = section.split('\\')[-2]
            # state_list = ['dataset/row\\metal_nut\\train\\good']
            state_list = glob(os.path.join(section, '*'))

            for state in state_list:
                # state_name: ex. 'good'
                state_name = state.split('\\')[-1]

                resize_state_path = os.path.join(resize_path, material_name, section_name, state_name)
                if not os.path.exists(resize_state_path):
                    os.makedirs(resize_state_path)

                [resize_img(in_path=img, out_path=resize_state_path, size=size, mode=args.mode) for img in glob(os.path.join(state, '*'))]

"""
numpy配列に変換
"""
def trans_main(args):
    # material_list = ['dataset/row\\metal_nut']
    material_list = glob('dataset\\{}\\*'.format(args.wrapper_name))
    for material in material_list:
        # material_name: ex. 'metal_nut'
        material_name = material.split('\\')[-1]
        # section_list = ['dataset/row\\metal_nut\\ground_truth\\', 'dataset/row\\metal_nut\\test\\', 'dataset/row\\metal_nut\\train\\']
        section_list = glob(os.path.join(material, '*\\'))

        for section in section_list:
            # section_name: ex. 'train'
            section_name = section.split('\\')[-2]
            # state_list = ['dataset/row\\metal_nut\\train\\good']
            state_list = glob(os.path.join(section, '*'))

            for state in state_list:
                # state_name: ex. 'good'
                state_name = state.split('\\')[-1]

                save_name = '{}_{}'.format(section_name, state_name)
                transform_to_numpy(wrapper_path=state, out_path=material, save_name=save_name, isMiniMax=args.isMM)


def check_numpy(pkl):
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    print('# data.shape: {}'.format(data.shape))
    print('# data.dtype: {}'.format(data.dtype))
    print('# data: {}'.format(data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Util Resize")
    parser.add_argument('--size', '-s', default='128,128', type=str)
    parser.add_argument('--mode', '-m', default='LANCZOS', type=str)
    parser.add_argument('--wrapper_name', '-wn', default='resize_128_128_LANCZOS', type=str)
    parser.add_argument('--isMM', '-is', default=False, type=bool)
    args = parser.parse_args()

    # resize_main(args)
    # trans_main(args)
    check_numpy('dataset/resize_128_128_LANCZOS/metal_nut/train_good.pkl')
    check_numpy('dataset/resize_128_128_LANCZOS/metal_nut/ground_truth_bent.pkl')