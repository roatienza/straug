"""
Curved, distorted and stretched text styles are found in natural scenes.
Warp simulates these data augmentation functions:
1) Curve,
2) Distort
3) Stretch

Stretch can also simulate contractions.
All functions use Thin-Splie-Plate (TPS) algorithm.

Copyright 2021 Rowel Atienza
"""

import cv2
import numpy as np
from PIL import Image, ImageOps


class Stretch:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()

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
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([p, p])
        srcpt.append([p, h - p])
        srcpt.append([p, h_50])
        x = self.rng.uniform(0, frac) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt.append([p + x, p])
        dstpt.append([p + x, h - p])
        dstpt.append([p + x, h_50])

        # 2nd left-most
        srcpt.append([p + w_33, p])
        srcpt.append([p + w_33, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        dstpt.append([p + w_33 + x, p])
        dstpt.append([p + w_33 + x, h - p])

        # 3rd left-most
        srcpt.append([p + w_66, p])
        srcpt.append([p + w_66, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        dstpt.append([p + w_66 + x, p])
        dstpt.append([p + w_66 + x, h - p])

        # right-most
        srcpt.append([w - p, p])
        srcpt.append([w - p, h - p])
        srcpt.append([w - p, h_50])
        x = self.rng.uniform(-frac, 0) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt.append([w - p + x, p])
        dstpt.append([w - p + x, h - p])
        dstpt.append([w - p + x, h_50])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img


class Distort:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()

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
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # top pts
        srcpt.append([p, p])
        x = self.rng.uniform(0, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + x, p + y])

        srcpt.append([p + w_33, p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + w_33 + x, p + y])

        srcpt.append([p + w_66, p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + w_66 + x, p + y])

        srcpt.append([w - p, p])
        x = self.rng.uniform(-frac, 0) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([w - p + x, p + y])

        # bottom pts
        srcpt.append([p, h - p])
        x = self.rng.uniform(0, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + x, h - p + y])

        srcpt.append([p + w_33, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + w_33 + x, h - p + y])

        srcpt.append([p + w_66, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + w_66 + x, h - p + y])

        srcpt.append([w - p, h - p])
        x = self.rng.uniform(-frac, 0) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([w - p + x, h - p + y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img


class Curve:
    def __init__(self, square_side=224, rng=None):
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.side = square_side
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        orig_w, orig_h = img.size

        if orig_h != self.side or orig_w != self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        isflip = self.rng.uniform(0, 1) > 0.5
        if isflip:
            img = ImageOps.flip(img)
            # img = TF.vflip(img)

        img = np.asarray(img)
        w = self.side
        h = self.side
        w_25 = 0.25 * w
        w_50 = 0.50 * w
        w_75 = 0.75 * w

        b = [1.1, .95, .8]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        rmin = b[index]

        r = self.rng.uniform(rmin, rmin + .1) * h
        x1 = (r ** 2 - w_50 ** 2) ** 0.5
        h1 = r - x1

        t = self.rng.uniform(0.4, 0.5) * h

        w2 = w_50 * t / r
        hi = x1 * t / r
        h2 = h1 + hi

        sinb_2 = ((1 - x1 / r) / 2) ** 0.5
        cosb_2 = ((1 + x1 / r) / 2) ** 0.5
        w3 = w_50 - r * sinb_2
        h3 = r - r * cosb_2

        w4 = w_50 - (r - t) * sinb_2
        h4 = r - (r - t) * cosb_2

        w5 = 0.5 * w2
        h5 = h1 + 0.5 * hi
        h_50 = 0.50 * h

        srcpt = [(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h),
                 (0, h_50), (w, h_50)]
        dstpt = [(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4),
                 (w - w4, h4), (w5, h5), (w - w5, h5)]

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if isflip:
            # img = TF.vflip(img)
            img = ImageOps.flip(img)
            rect = (0, self.side // 2, self.side, self.side)
        else:
            rect = (0, 0, self.side, self.side // 2)

        img = img.crop(rect)
        img = img.resize((orig_w, orig_h), Image.BICUBIC)
        return img
