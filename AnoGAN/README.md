# AnoGAN (Anomaly GAN)

![](https://img.shields.io/badge/chainer-5.4.0-red.svg)

We implemented [AnoGAN](https://arxiv.org/pdf/1703.05921.pdf) by chainer.  
DCGAN was used in this paper but we used ProgressiveGAN to deal with larger scale images.  

## Quick Start

```console
# GAN training
$ python train.py
# Anomaly training and defect segmentation
$ python detect.py
```

## Related Works

* Progressive growing of gans for improved quality, stability, and variation | Karras, Tero, et al | **[ICLR 2018]** | <a href="https://arxiv.org/pdf/1710.10196.pdf?__hstc=200028081.1bb630f9cde2cb5f07430159d50a3c91.1524009600081.1524009600082.1524009600083.1&__hssc=200028081.1.1524009600084&__hsfp=1773666937" rel="nofollow"><code>[pdf]</code></a>
