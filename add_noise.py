import random
import numpy as np
from PIL import Image


class add_saltPepper_noise(object):
    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1.0 - Nd
            mask = np.random.choice((0,1,2), size=(h,w,1), p=[Nd/2.0, Nd/2.0, Sd])
            mask = np.repeat(mask, c, axis=2)
            img[mask==0] = 0
            img[mask==1] = 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img

class add_gaussian_noise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=0.5):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(
                loc=self.mean, 
                scale = self.variance, 
                size=(h, w, 1)
            )
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
