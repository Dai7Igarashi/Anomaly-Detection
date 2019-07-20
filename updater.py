# coding: UTF-8

import numpy as np
import math
from data_preprocess import data_augmentation

import chainer
import chainer.links as L
import chainer.functions as F

from visualize import sample_generate, sample_generate_light

def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] *= (1 - tau)
        target_params[param_name].data[:] += tau * param.data

    # Soft-copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] *= (1 - tau)
            target_bn.avg_mean[:] += tau * link.avg_mean
            target_bn.avg_var[:] *= (1 - tau)
            target_bn.avg_var[:] += tau * link.avg_var

def updater(
        iterator_train, epoch, max_epoch, step, xp, save_path,
        ano, gen, dis, gen_s, opt_gen, opt_dis,
        n_dis, lam, gamma, smoothing
):
    for i in range(n_dis):
        for train_batch in iterator_train(shuffle=True):
            batch = data_augmentation(np.asarray(train_batch['images']), resize=(512,512))
            batchsize = len(batch)
            batch = (batch - 127.5) / 127.5
            x_real = chainer.Variable(xp.asarray(batch))

            stage = 2*step + (epoch / (max_epoch/2 + 1))

            if math.floor(stage)%2==0:
                reso = min(512, 4 * 2**(((math.floor(stage)+1)//2)))
                scale = max(1, 512//reso)
                if scale>1:
                    x_real = F.average_pooling_2d(x_real, scale, scale, 0)
            else:
                alpha = stage - math.floor(stage)
                reso_low = min(512, 4 * 2**(((math.floor(stage))//2)))
                reso_high = min(512, 4 * 2**(((math.floor(stage)+1)//2)))
                scale_low = max(1, 512//reso_low)
                scale_high = max(1, 512//reso_high)
                if scale_low>1:
                    x_real_low = F.unpooling_2d(
                        F.average_pooling_2d(x_real, scale_low, scale_low, 0),
                        2, 2, 0, outsize=(reso_high, reso_high))
                    x_real_high = F.average_pooling_2d(x_real, scale_high, scale_high, 0)
                    x_real = (1-alpha)*x_real_low + alpha*x_real_high

            y_real = dis(x_real, stage=stage)

            z = chainer.Variable(xp.asarray(ano.make_hidden(batchsize)))
            z = ano(z, batchsize)
            x_fake = gen(z, stage=stage)
            y_fake = dis(x_fake, stage=stage)

            x_fake.unchain_backward()

            eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            x_mid_v = chainer.Variable(x_mid.data)
            y_mid = F.sum(dis(x_mid_v, stage=stage))

            dydx, = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)
            dydx = F.sqrt(F.sum(dydx * dydx, axis=(1, 2, 3)))
            loss_gp = lam * F.mean_squared_error(dydx, gamma * xp.ones_like(dydx.data)) * (1.0 / gamma ** 2)

            loss_dis = F.sum(-y_real) / batchsize
            loss_dis += F.sum(y_fake) / batchsize

            loss_dis += 0.001 * F.sum(y_real ** 2) / batchsize

            loss_dis_total = loss_dis + loss_gp
            dis.cleargrads()
            loss_dis_total.backward()
            opt_dis.update()
            loss_dis_total.unchain_backward()

            z = chainer.Variable(xp.asarray(ano.make_hidden(batchsize)))
            z = ano(z, batchsize)
            x_fake = gen(z, stage=stage)
            y_fake = dis(x_fake, stage=stage)
            loss_gen = F.sum(-y_fake) / batchsize
            gen.cleargrads()
            loss_gen.backward()
            opt_gen.update()

            soft_copy_param(gen_s, gen, 1.0 - smoothing)

            print(
                '# {:>5}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('step', 'epoch', 'stage', 'loss_dis',
                                                                         'loss_gp', 'loss_gen', 'g'))
            print('# {:>5}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
                step, epoch, stage, str(loss_dis.array), str(loss_gp.array), str(loss_gen.array), str(F.mean(dydx).array)
            ))

            if epoch % 50 == 0:
                sample_generate_light(ano, gen, stage, save_path)
            if epoch % 100 == 0:
                sample_generate(ano, gen, stage, save_path)