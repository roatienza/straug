"""
When viewing natural scenes, perfect horizontal frontal alignment
is seldom achieved. Almost always there is some degree of rotation 
and perspective transformation in the text image. 
Text may not also be perfectly centered. 
Translation along x and/or y coordinates is common. 
Furthermore, text can be found in varying sizes. 
To simulate these real-world scenarios, Geometry offers:
1) Perspective, 
2) Shrink and 
3) Rotate

Copyright 2021 Rowel Atienza
"""

import cv2
import numpy as np
from PIL import Image


class Shrink:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.translateXAbs = TranslateXAbs(self.rng)
        self.translateYAbs = TranslateYAbs(self.rng)

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = np.asarray(img)
        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0

        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([p, p])
        srcpt.append([p, h - p])
        x = self.rng.uniform(frac - .1, frac) * w_33
        y = self.rng.uniform(frac - .1, frac) * h_50
        dstpt.append([p + x, p + y])
        dstpt.append([p + x, h - p - y])

        # 2nd left-most 
        srcpt.append([p + w_33, p])
        srcpt.append([p + w_33, h - p])
        dstpt.append([p + w_33, p + y])
        dstpt.append([p + w_33, h - p - y])

        # 3rd left-most 
        srcpt.append([p + w_66, p])
        srcpt.append([p + w_66, h - p])
        dstpt.append([p + w_66, p + y])
        dstpt.append([p + w_66, h - p - y])

        # right-most 
        srcpt.append([w - p, p])
        srcpt.append([w - p, h - p])
        dstpt.append([w - p - x, p + y])
        dstpt.append([w - p - x, h - p - y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if self.rng.uniform(0, 1) < 0.5:
            img = self.translateXAbs(img, val=x)
        else:
            img = self.translateYAbs(img, val=y)

        return img


class Rotate:
    def __init__(self, square_side=224, rng=None):
        self.side = square_side
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, iscurve=False, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size

        if h != self.side or w != self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        b = [20., 40, 60]
        if mag < 0 or mag >= len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = self.rng.uniform(rotate_angle - 20, rotate_angle)
        if self.rng.uniform(0, 1) < 0.5:
            angle = -angle

        img = img.rotate(angle=angle, resample=Image.BICUBIC, expand=not iscurve)
        img = img.resize((w, h), Image.BICUBIC)

        return img


class Perspective:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size

        # upper-left, upper-right, lower-left, lower-right
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        # low = 0.3

        b = [.1, .2, .3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        low = b[index]

        high = 1 - low
        if self.rng.uniform(0, 1) > 0.5:
            topright_y = self.rng.uniform(low, low + .1) * h
            bottomright_y = self.rng.uniform(high - .1, high) * h
            dest = np.float32([[0, 0], [w, topright_y], [0, h], [w, bottomright_y]])
        else:
            topleft_y = self.rng.uniform(low, low + .1) * h
            bottomleft_y = self.rng.uniform(high - .1, high) * h
            dest = np.float32([[0, topleft_y], [w, 0], [0, bottomleft_y], [w, h]])
        M = cv2.getPerspectiveTransform(src, dest)
        img = np.asarray(img)
        img = cv2.warpPerspective(img, M, (w, h))
        img = Image.fromarray(img)

        return img


class TranslateX:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        b = [.03, .06, .09]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = self.rng.uniform(v - 0.03, v)

        v = v * img.size[0]
        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateY:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        b = [.07, .14, .21]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = self.rng.uniform(v - 0.07, v)

        v = v * img.size[1]
        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


class TranslateXAbs:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, val=0, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        v = self.rng.uniform(0, val)

        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateYAbs:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, val=0, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        v = self.rng.uniform(0, val)

        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
