# coding: UTF-8

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

class AnoNet(chainer.Chain):
    def __init__(self, input=512):
        super(AnoNet, self).__init__()
        w = chainer.initializers.Normal(1.0)
        self.input = input
        with self.init_scope():
            self.l0 = L.Linear(input, input, initialW=w)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.input)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.input + 1e-8)
        return z

    def __call__(self, z, batchsize):
        z = F.leaky_relu(self.l0(z))
        return F.reshape(z, (batchsize, self.input, 1, 1))