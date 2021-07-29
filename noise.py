"""
Noise is common in natural images. 
Noise supports:
1) GaussianNoise, 
2) ShotNoise, 
3) ImpulseNoise and 
4) SpeckleNoise.

Noise algorithms from https://github.com/hendrycks/robustness
Hacked together for STR by: Rowel Atienza
"""

import numpy as np
import skimage as sk
from PIL import Image


class GaussianNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(.08, .38)
        b = [.06, 0.09, 0.12]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + 0.03)
        img = np.asarray(img) / 255.
        img = np.clip(img + self.rng.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ShotNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        # Create a dedicated rng for the Poisson noise
        self.noise = np.random.Generator(self.rng.bit_generator.jumped())

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(3, 60)
        b = [13, 8, 3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + 7)
        img = np.asarray(img) / 255.
        img = np.clip(self.noise.poisson(img * c) / float(c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ImpulseNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(.03, .27)
        b = [.03, .07, .11]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + .04)
        # sk.util.random_noise() uses legacy np.random.* functions.
        # We can't pass an rng instance so we specify the seed instead.
        # np.random.seed() accepts 32-bit integers only,
        # generate 4 to simulate a 128-bit state/seed.
        s = self.rng.integers(2 ** 32, size=4)
        img = sk.util.random_noise(np.asarray(img) / 255., mode='s&p', seed=s, amount=c) * 255
        return Image.fromarray(img.astype(np.uint8))


class SpeckleNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(.15, .6)
        b = [.15, .2, .25]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + .05)
        img = np.asarray(img) / 255.
        img = np.clip(img + img * self.rng.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))
