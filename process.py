"""
All other image transformations used in object recognition data 
augmentation literature that may be applicable in STR can be
found here:
1) Posterize
2) Solarize,
3) Invert, 
4) Equalize, 
5) AutoContrast, 
6) Sharpness and 
7) Color.

Based on AutoAugment and FastAugment:
    https://github.com/kakaobrain/fast-autoaugment

Hacked together for STR by: Rowel Atienza
"""


from PIL import Image
import PIL.ImageOps, PIL.ImageEnhance
import numpy as np

class Posterize:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        c = [1, 3, 6]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        bit = np.random.randint(c, c+2)
        img = PIL.ImageOps.posterize(img, bit)

        return img


class Solarize:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        c = [64, 128, 192]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        thresh = np.random.randint(c, c+64)
        img = PIL.ImageOps.solarize(img, thresh)

        return img

class Invert:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        img = PIL.ImageOps.invert(img)

        return img

    
class Equalize:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        mg = PIL.ImageOps.equalize(img)

        return img


class AutoContrast:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        mg = PIL.ImageOps.autocontrast(img)

        return img


class Sharpness:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        c = [.1, .7, 1.3]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c+.6)
        img = PIL.ImageEnhance.Sharpness(img).enhance(magnitude)

        return img


class Color:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        c = [.1, .7, 1.3]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c+.6)
        img = PIL.ImageEnhance.Color(img).enhance(magnitude)

        return img


