# coding: UTF-8

import argparse
import os
import sys
import numpy as np
import time
import logging
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

import chainer
import chainer.functions as F
from chainer.backends.cuda import to_cpu
from chainer.computational_graph import build_computational_graph

from anonet import AnoNet
from pggan import Generator, Discriminator
from updater import updater

class Iterator(object):
    def __init__(self, images, batchsize):
        self.images = images
        self.batchsize = batchsize

    @staticmethod
    def get_index(images, shuffle):
        index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(index)
        return index

    @staticmethod
    def get_minibatch(images, ref):
        output = {'images': images[ref]}
        return output

    def __call__(self, shuffle=False):
        index = self.get_index(self.images, shuffle)

        for start in range(0, len(self.images), self.batchsize):
            ref = index[start:start+self.batchsize]
            yield self.get_minibatch(self.images, ref)

# logger設定用関数
def set_logger(logger_name, save_path):
    loggger = logging.getLogger(logger_name)
    loggger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('{}/{}.txt'.format(save_path, logger_name), 'w')
    loggger.addHandler(file_handler)
    return loggger

def save_model(ano, gen, gen_s, dis, gpu, path):
    if gpu >= 0:
        ano.to_cpu()
        gen.to_cpu()
        gen_s.to_cpu()
        dis.to_cpu()
        chainer.serializers.save_npz(os.path.join(path, 'ano_net.model'), ano)
        chainer.serializers.save_npz(os.path.join(path, 'generator.model'), gen)
        chainer.serializers.save_npz(os.path.join(path, 'generator_smooth.model'), gen_s)
        chainer.serializers.save_npz(os.path.join(path, 'discriminator.model'), dis)
        ano.to_gpu()
        gen.to_gpu()
        gen_s.to_gpu()
        dis.to_gpu()
    else:
        chainer.serializers.save_npz(os.path.join(path, 'ano_net.model'), ano)
        chainer.serializers.save_npz(os.path.join(path, 'generator.model'), gen)
        chainer.serializers.save_npz(os.path.join(path, 'generator_smooth.model'), gen_s)
        chainer.serializers.save_npz(os.path.join(path, 'discriminator.model'), dis)

def main():
    parser = argparse.ArgumentParser(description='Train PGGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--total_step', '-ts', type=int, default=8)
    parser.add_argument('--step0_epoch', '-s0e', type=int, default=200, help='generate (4x4) image step')
    parser.add_argument('--step1_epoch', '-s1e', type=int, default=200, help='generate (8x8) image step')
    parser.add_argument('--step2_epoch', '-s2e', type=int, default=300, help='generate (16x16) image step')
    parser.add_argument('--step3_epoch', '-s3e', type=int, default=300, help='generate (32x32) image step')
    parser.add_argument('--step4_epoch', '-s4e', type=int, default=300, help='generate (64x64) image step')
    parser.add_argument('--step5_epoch', '-s5e', type=int, default=500, help='generate (128x128) image step')
    parser.add_argument('--step6_epoch', '-s6e', type=int, default=500, help='generate (256x256) image step')
    parser.add_argument('--step7_epoch', '-s7e', type=int, default=500, help='generate (512x512) image step')
    parser.add_argument('--load_path', '-lp', type=str, default='', help='.\\result\\1907201234_123456\\step_3')

    parser.add_argument('--n_dis', type=int, default=1,
                        help='number of discriminator update per generator update')
    parser.add_argument('--lam', type=float, default=10,
                        help='gradient penalty')
    parser.add_argument('--gamma', type=float, default=750,
                        help='gradient penalty')
    parser.add_argument('--pooling_comp', type=float, default=1.0,
                        help='compensation')
    parser.add_argument('--generator_smoothing', type=float, default=0.999)

    args = parser.parse_args()

    # resultフォルダ作成
    _result_dir = '.\\result'
    if not os.path.exists(_result_dir):
        print('*** Create result folder ***')
        os.makedirs(_result_dir)

    now_time = time.strftime('%y%m%d_%H%M%S', time.localtime())
    # loggerおよびmodel保存用フォルダ作成
    _log_save_path = os.path.join(_result_dir, now_time)
    os.makedirs(_log_save_path)

    for i in range(args.total_step):
        os.makedirs(os.path.join(_log_save_path, 'step{}'.format(i)))

    logger_names = ['misc']
    loggers = {}
    for logger_name in logger_names:
        loggers[logger_name] = set_logger(logger_name, _log_save_path)

    # hyper params
    loggers['misc'].debug('# batchsize: {}'.format(args.batchsize))
    loggers['misc'].debug('# step0_epoch: {}'.format(args.step0_epoch))
    loggers['misc'].debug('# step1_epoch: {}'.format(args.step1_epoch))
    loggers['misc'].debug('# step2_epoch: {}'.format(args.step2_epoch))
    loggers['misc'].debug('# step3_epoch: {}'.format(args.step3_epoch))
    loggers['misc'].debug('# step4_epoch: {}'.format(args.step4_epoch))
    loggers['misc'].debug('# step5_epoch: {}'.format(args.step5_epoch))
    loggers['misc'].debug('# step6_epoch: {}'.format(args.step6_epoch))
    loggers['misc'].debug('# step7_epoch: {}'.format(args.step7_epoch))
    loggers['misc'].debug('# load_path: {}'.format(args.load_path))
    loggers['misc'].debug('# n_dis: {}'.format(args.n_dis))
    loggers['misc'].debug('# lam: {}'.format(args.lam))
    loggers['misc'].debug('# gamma: {}'.format(args.gamma))
    loggers['misc'].debug('# pooling_comp: {}'.format(args.pooling_comp))
    loggers['misc'].debug('# generator_smoothing: {}'.format(args.generator_smoothing))
    loggers['misc'].debug('-----')

    # load dataset
    good_path = glob('dataset\\train\\good\\*')
    datas = []
    for path in good_path:
        img = Image.open(path)
        img = img.convert('L')
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        datas.append(img)
    """
    good_path = 'dataset\\good\\row'
    datas = []
    for path in glob(os.path.join(good_path, '*')):
        for img_path in glob(os.path.join(path, '*.tif')):
            img = Image.open(img_path)
            img = np.asarray(img)
            H, W = img.shape
            if H == 1236 and W == 1626:
                img = np.expand_dims(img, axis=0)
                datas.append(img)
    """
    train_dataset = np.asarray(datas)
    print('# train_dataset: {}'.format(train_dataset.shape))
    loggers['misc'].debug('# train_dataset: {}'.format(train_dataset.shape))

    # model呼び出し
    ano_net = AnoNet()
    generator = Generator()
    generator_smooth = Generator()
    discriminator = Discriminator()

    # cpu_or_gpu
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        ano_net.to_gpu()
        generator.to_gpu()
        generator_smooth.to_gpu()
        discriminator.to_gpu()
    xp = chainer.backends.cuda.cupy if args.gpu >= 0 else np

    if args.load_path != '':
        chainer.serializers.load_npz(os.path.join(args.load_path, 'ano_net.model'), ano_net)
        chainer.serializers.load_npz(os.path.join(args.load_path, 'generator.model'), generator)
        chainer.serializers.load_npz(os.path.join(args.load_path, 'generator_smooth.model'), generator_smooth)
        chainer.serializers.load_npz(os.path.join(args.load_path, 'discriminator.model'), discriminator)

    # optimizer設定
    def make_optimizer(model, alpha=0.001, beta1=0.0, beta2=0.99):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(generator)
    opt_dis = make_optimizer(discriminator)

    # 学習ループ
    iterator_train = Iterator(train_dataset, args.batchsize)

    # STEP0: generate (4x4) image
    for epoch in range(args.step0_epoch):
        updater(iterator_train, epoch, max_epoch=args.step0_epoch, step=0, xp=xp, save_path='{}\\step0'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step0'.format(_log_save_path))

    # STEP1: generate (8x8) image
    for epoch in range(args.step1_epoch):
        updater(iterator_train, epoch, max_epoch=args.step1_epoch,  step=1, xp=xp, save_path='{}\\step1'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step1'.format(_log_save_path))

    # STEP2: generate (16x16) image
    for epoch in range(args.step2_epoch):
        updater(iterator_train, epoch, max_epoch=args.step2_epoch,  step=2, xp=xp, save_path='{}\\step2'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step2'.format(_log_save_path))

    # STEP3: generate (32x32) image
    for epoch in range(args.step3_epoch):
        updater(iterator_train, epoch, max_epoch=args.step3_epoch,  step=3, xp=xp, save_path='{}\\step3'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step3'.format(_log_save_path))

    # STEP4: generate (64x64) image
    for epoch in range(args.step4_epoch):
        updater(iterator_train, epoch, max_epoch=args.step4_epoch,  step=4, xp=xp, save_path='{}\\step4'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step4'.format(_log_save_path))

    # STEP5: generate (128x128) image
    for epoch in range(args.step5_epoch):
        updater(iterator_train, epoch, max_epoch=args.step5_epoch,  step=5, xp=xp, save_path='{}\\step5'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step5'.format(_log_save_path))

    # STEP6: generate (256x256) image
    for epoch in range(args.step6_epoch):
        updater(iterator_train, epoch, max_epoch=args.step6_epoch,  step=6, xp=xp, save_path='{}\\step6'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step6'.format(_log_save_path))

    # STEP7: generate (512x512) image
    for epoch in range(args.step7_epoch):
        updater(iterator_train, epoch, step=7, max_epoch=args.step7_epoch,  xp=xp, save_path='{}\\step7'.format(_log_save_path),
                ano=ano_net, gen=generator, dis=discriminator, gen_s=generator_smooth, opt_gen=opt_gen, opt_dis=opt_dis,
                n_dis=args.n_dis, lam=args.lam, gamma=args.gamma, smoothing=args.generator_smoothing)
    save_model(ano=ano_net, gen=generator, gen_s=generator_smooth, dis=discriminator, gpu=args.gpu,
               path='{}\\step7'.format(_log_save_path))


if __name__ == '__main__':
    main()