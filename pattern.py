"""
Inspired by GridMask, 5 grid patterns that mask out certain regions
from the image while ensuring that text symbols are still readable. 
1) Grid, 
2) VGrid, 
3) HGrid, 
4) RectGrid 
and 
5) EllipseGrid

Copyright 2021 Rowel Atienza
"""

import numpy as np
from PIL import ImageDraw


class VGrid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        if copy:
            img = img.copy()
        w, h = img.size

        if mag < 0 or mag > max_width:
            line_width = self.rng.integers(1, max_width)
            image_stripe = self.rng.integers(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = w // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            x = image_stripe * i + line_width * (i - 1)
            draw.line([(x, 0), (x, h)], width=line_width, fill='black')

        return img


class HGrid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        if copy:
            img = img.copy()
        w, h = img.size
        if mag < 0 or mag > max_width:
            line_width = self.rng.integers(1, max_width)
            image_stripe = self.rng.integers(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = h // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            y = image_stripe * i + line_width * (i - 1)
            draw.line([(0, y), (w, y)], width=line_width, fill='black')

        return img


class Grid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = VGrid(self.rng)(img, copy=True, mag=mag)
        img = HGrid(self.rng)(img, copy=False, mag=mag)
        return img


class RectGrid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, isellipse=False, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = img.copy()
        w, h = img.size
        line_width = 1
        image_stripe = 3 - mag  # self.rng.integers(2, 6)
        offset = 4 if isellipse else 1
        n_lines = ((h // 2) // (line_width + image_stripe)) + offset
        draw = ImageDraw.Draw(img)
        x_center = w // 2
        y_center = h // 2
        for i in range(1, n_lines):
            dx = image_stripe * i + line_width * (i - 1)
            dy = image_stripe * i + line_width * (i - 1)
            x1 = x_center - (dx * w // h)
            y1 = y_center - dy
            x2 = x_center + (dx * w / h)
            y2 = y_center + dy
            if isellipse:
                draw.ellipse([(x1, y1), (x2, y2)], width=line_width, outline='black')
            else:
                draw.rectangle([(x1, y1), (x2, y2)], width=line_width, outline='black')

        return img


class EllipseGrid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = RectGrid(self.rng)(img, isellipse=True, mag=mag, prob=prob)
        return img
