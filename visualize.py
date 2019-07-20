import os

import numpy as np
from PIL import Image

import chainer
from chainer.backends.cuda import to_cpu
from chainer import Variable

def sample_generate_light(ano, gen, stage, save_path, rows=5, cols=5, seed=0):
    np.random.seed(seed)
    n_images = rows * cols
    xp = ano.xp
    z = Variable(xp.asarray(ano.make_hidden(n_images)))
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        z = ano(z, n_images)
        x = gen(z, stage=stage)
    x = to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 1, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W))

    preview_dir = os.path.join(save_path, 'preview')
    preview_path = os.path.join(preview_dir, 'image_latest.png')
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)


def sample_generate(ano, gen, stage, save_path, rows=10, cols=10, seed=0):
    np.random.seed(seed)
    n_images = rows * cols
    xp = ano.xp
    z = Variable(xp.asarray(ano.make_hidden(n_images)))
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        z = ano(z, n_images)
        x = gen(z, stage=stage)
    x = to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, h, w = x.shape
    x = x.reshape((rows, cols, 1, h, w))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * h, cols * w))

    preview_dir = os.path.join(save_path, 'preview')
    preview_path = os.path.join(preview_dir, 'image{:0>8}.png'.format(stage))
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)
