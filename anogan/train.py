# coding: UTF-8

"""AnoGAN example
using: metal_nut data
"""

import argparse

def train(args):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AnoGAN for Chainer")
    parser.add_argument('--batchsize', '-b', default=64, type=int)
    parser.add_argument('--epoch', '-e', default=25, type=int)
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--optimizer', '-opt', default='Adam', choices=['Adam'])
    parser.add_argument('--adam_alpha', '-al', default=0.0002, type=float)
    parser.add_argument('--adam_beta1', '-ab', default=0.5, type=float)

    args = parser.parse_args()
    train(args)