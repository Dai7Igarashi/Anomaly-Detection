import os
import math

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

def feature_vector_normalization(x, eps=1e-8):
    # x: (B, C, H, W)
    alpha = 1.0 / F.sqrt(F.mean(x*x, axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.data.shape) * x


class EqualizedConv2d(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/(in_ch*ksize**2))
        super(EqualizedConv2d, self).__init__()
        with self.init_scope():
            self.c = L.Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w)
    def __call__(self, x):
        return self.c(self.inv_c * x)

class EqualizedLinear(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/in_ch)
        super(EqualizedLinear, self).__init__()
        with self.init_scope():
            self.c = L.Linear(in_ch, out_ch, initialW=w)
    def __call__(self, x):
        return self.c(self.inv_c * x)

class EqualizedDeconv2d(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/(in_ch))
        super(EqualizedDeconv2d, self).__init__()
        with self.init_scope():
            self.c = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w)
    def __call__(self, x):
        return self.c(self.inv_c * x)

def minibatch_std(x):
    m = F.mean(x, axis=0, keepdims=True)
    v = F.mean((x - F.broadcast_to(m, x.shape))*(x - F.broadcast_to(m, x.shape)), axis=0, keepdims=True)
    std = F.mean(F.sqrt(v + 1e-8), keepdims=True)
    std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    return F.concat([x, std], axis=1)

class GeneratorBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        super(GeneratorBlock, self).__init__()
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)
    def __call__(self, x):
        h = F.unpooling_2d(x, 2, 2, 0, outsize=(x.shape[2]*2, x.shape[3]*2))
        h = F.leaky_relu(feature_vector_normalization(self.c0(h)))
        h = F.leaky_relu(feature_vector_normalization(self.c1(h)))
        return h


class Generator(chainer.Chain):
    def __init__(self, n_hidden=512, ch=512, max_stage=14):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.max_stage = max_stage
        with self.init_scope():
            #self.c0 = EqualizedDeconv2d(n_hidden, ch, 4, 1, 0)
            # out0 generate (4x4)
            self.c0 = EqualizedConv2d(n_hidden, ch, 4, 1, 3)
            self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1)
            self.out0 = EqualizedConv2d(ch, 1, 1, 1, 0)
            # out1 generate (8x8)
            self.b1 = GeneratorBlock(ch, ch)
            self.out1 = EqualizedConv2d(ch, 1, 1, 1, 0)
            # out2 generate (16x16)
            self.b2 = GeneratorBlock(ch, ch)
            self.out2 = EqualizedConv2d(ch, 1, 1, 1, 0)
            # out3 generate (32x32)
            self.b3 = GeneratorBlock(ch, ch//2)
            self.out3 = EqualizedConv2d(ch//2, 1, 1, 1, 0)
            # out4 generate (64x64)
            self.b4 = GeneratorBlock(ch//2, ch//4)
            self.out4 = EqualizedConv2d(ch//4, 1, 1, 1, 0)
            # out5 generate (128x128)
            self.b5 = GeneratorBlock(ch//4, ch//8)
            self.out5 = EqualizedConv2d(ch//8, 1, 1, 1, 0)
            # out6 generate (256x256)
            self.b6 = GeneratorBlock(ch//8, ch//16)
            self.out6 = EqualizedConv2d(ch//16, 1, 1, 1, 0)
            # out7 generate (512x512)
            self.b7 = GeneratorBlock(ch//16, ch//32)
            self.out7 = EqualizedConv2d(ch//32, 1, 1, 1, 0)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)) \
            .astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z

    def __call__(self, z, stage):
        # stage0: c0->c1->out0
        # stage1: c0->c1-> (1-a)*(up->out0) + (a)*(b1->out1)
        # stage2: c0->c1->b1->out1
        # stage3: c0->c1->b1-> (1-a)*(up->out1) + (a)*(b2->out2)
        # stage4: c0->c1->b2->out2
        # ...

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = F.reshape(z,(len(z), self.n_hidden, 1, 1))
        h = F.leaky_relu(feature_vector_normalization(self.c0(h)))
        h = F.leaky_relu(feature_vector_normalization(self.c1(h)))

        for i in range(1, int(stage//2+1)):
            h = getattr(self, "b%d"%i)(h)

        if int(stage)%2==0:
            out = getattr(self, "out%d"%(stage//2))
            x = out(h)
        else:
            out_prev = getattr(self, "out%d"%(stage//2))
            out_curr = getattr(self, "out%d"%(stage//2+1))
            b_curr = getattr(self, "b%d"%(stage//2+1))

            x_0 = out_prev(F.unpooling_2d(h, 2, 2, 0, outsize=(2*h.shape[2], 2*h.shape[3])))
            x_1 = out_curr(b_curr(h))
            x = (1.0-alpha)*x_0 + alpha*x_1

        if chainer.configuration.config.train:
            return x
        else:
            scale = int(512 // x.data.shape[2])
            return F.unpooling_2d(x, scale, scale, 0, outsize=(512,512))


class DiscriminatorBlock(chainer.Chain):
    # conv-conv-downsample
    def __init__(self, in_ch, out_ch, pooling_comp):
        super(DiscriminatorBlock, self).__init__()
        self.pooling_comp = pooling_comp
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, in_ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
    def __call__(self, x):
        h = F.leaky_relu((self.c0(x)))
        h = F.leaky_relu((self.c1(h)))
        h = self.pooling_comp * F.average_pooling_2d(h, 2, 2, 0)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, ch=512, max_stage=14, pooling_comp=1.0):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage
        self.pooling_comp = pooling_comp # compensation of ave_pool is 0.5-Lipshitz
        with self.init_scope():
            # in7: (512x512)
            self.in7 = EqualizedConv2d(1, ch//32, 1, 1, 0)
            self.b7 = DiscriminatorBlock(ch//32, ch, pooling_comp)
            # in6: (256x256)
            self.in6 = EqualizedConv2d(1, ch//16, 1, 1, 0)
            self.b6 = DiscriminatorBlock(ch//16, ch, pooling_comp)
            # in5: (128x128)
            self.in5 = EqualizedConv2d(1, ch//8, 1, 1, 0)
            self.b5 = DiscriminatorBlock(ch//8, ch, pooling_comp)
            # in4: (64x64)
            self.in4 = EqualizedConv2d(1, ch//4, 1, 1, 0)
            self.b4 = DiscriminatorBlock(ch//4, ch, pooling_comp)
            # in3: (32x32)
            self.in3 = EqualizedConv2d(1, ch//2, 1, 1, 0)
            self.b3 = DiscriminatorBlock(ch//2, ch, pooling_comp)
            # in2: (16x16)
            self.in2 = EqualizedConv2d(1, ch, 1, 1, 0)
            self.b2 = DiscriminatorBlock(ch, ch, pooling_comp)
            # in1: (8x8)
            self.in1 = EqualizedConv2d(1, ch, 1, 1, 0)
            self.b1 = DiscriminatorBlock(ch, ch, pooling_comp)
            # in0: (4x4)
            self.in0 = EqualizedConv2d(1, ch, 1, 1, 0)

            self.out0 = EqualizedConv2d(ch+1, ch, 3, 1, 1)
            self.out1 = EqualizedConv2d(ch, ch, 4, 1, 0)
            self.out2 = EqualizedLinear(ch, 1)

    def __call__(self, x, stage):
        # stage0: in0->m_std->out0_0->out0_1->out0_2
        # stage1: (1-a)*(down->in0) + (a)*(in1->b1) ->m_std->out0->out1->out2
        # stage2: in1->b1->m_std->out0_0->out0_1->out0_2
        # stage3: (1-a)*(down->in1) + (a)*(in2->b2) ->b1->m_std->out0->out1->out2
        # stage4: in2->b2->b1->m_std->out0->out1->out2
        # ...

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if int(stage)%2==0:
            fromRGB = getattr(self, "in%d"%(stage//2))
            h = F.leaky_relu(fromRGB(x))
        else:
            fromRGB0 = getattr(self, "in%d"%(stage//2))
            fromRGB1 = getattr(self, "in%d"%(stage//2+1))
            b1 = getattr(self, "b%d"%(stage//2+1))


            h0 = F.leaky_relu(fromRGB0(self.pooling_comp * F.average_pooling_2d(x, 2, 2, 0)))
            h1 = b1(F.leaky_relu(fromRGB1(x)))
            h = (1-alpha)*h0 + alpha*h1

        for i in range(int(stage // 2), 0, -1):
            h = getattr(self, "b%d" % i)(h)

        h = minibatch_std(h)
        h = F.leaky_relu((self.out0(h)))
        h = F.leaky_relu((self.out1(h)))
        return self.out2(h)