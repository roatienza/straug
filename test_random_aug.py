import argparse
import os
import numpy as np
from PIL import Image
import cv2

from straug.warp import Curve, Distort, Stretch
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.weather import Fog, Snow, Frost, Rain, Shadow

class Random_StrAug(object):
    def __init__(self, using_aug_types, prob_list = None):
        self.aug_list = []
        if 'warp' in using_aug_types :
            self.aug_list.append([Curve(), Distort(), Stretch()]) 
        if 'geometry' in using_aug_types :
            self.aug_list.append([Rotate(), Perspective(), Shrink(), TranslateX(), TranslateY()]) 
        if 'blur' in using_aug_types :
            self.aug_list.append([GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()]) 
        if 'noise' in using_aug_types :
            self.aug_list.append([GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()]) 
        if 'camera' in using_aug_types :
            self.aug_list.append([Contrast(), Brightness(), JpegCompression(), Pixelate()]) 
        if 'pattern' in using_aug_types :
            self.aug_list.append([VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]) 
        if 'process' in using_aug_types :
            self.aug_list.append([Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()]) 
        if 'weather' in using_aug_types :
            self.aug_list.append([Fog(), Snow(), Frost(), Rain(), Shadow()]) 
    
        self.mag_range = np.random.randint(-1, 3)
        if prob_list is None :
            self.prob_list = [0.5] * len(self.aug_list)
        else:
            assert len(self.aug_list) == len(prob_list), "The length of 'prob_list' must be the same as the number of augmentations used."
            self.prob_list = prob_list

    def __call__(self, img):
        for i in range(len(self.aug_list)):
            img = self.aug_list[i][np.random.randint(0, len(self.aug_list[i]))](img, mag = self.mag_range, prob = self.prob_list[i])

        return img
    
if __name__ == '__main__':
    random_StrAug_1 = Random_StrAug(using_aug_types = ['warp', 'geometry', 'blur', 'noise', 'camera', 'pattern', 'process', 'weather'],
                                  prob_list = [0.5, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    
    random_StrAug_2 = Random_StrAug(using_aug_types = ['warp', 'pattern', 'process', 'weather'],
                                  prob_list = [0.5, 0.3, 0.2, 0.5])
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="images/delivery.png", help='Load image file')
    parser.add_argument('--results', default="results", help='Folder for augmented image files')
    parser.add_argument('--gray', action='store_true', help='Convert to grayscale 1st')
    parser.add_argument('--width', default=200, type=int, help='Default image width')
    parser.add_argument('--height', default=64, type=int, help='Default image height')
    parser.add_argument('--seed', default=0, type=int, help='Random number generator seed')
    opt = parser.parse_args()
    os.makedirs(opt.results, exist_ok=True)

    img = Image.open(opt.image)
    img = img.resize((opt.width, opt.height))

    augmented_img_1 = random_StrAug_1(img)
    augmented_img_2 = random_StrAug_2(img)

    # Save images to compare before and after augmentation.
    result = cv2.cvtColor(np.hstack((np.array(img), np.array(augmented_img_1), np.array(augmented_img_2))), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(opt.results, opt.image.split('/')[-1].split('.')[0] + '_random_strAug.jpg'), result)