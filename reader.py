import torch
import torch.nn as nn
import torchvision
from PIL import Image
import torch.utils.data as data
import os
import cv2
from torchvision import transforms
from torchvision import transforms as T
from imgaug import augmenters as iaa
import numpy as np
class Reader(data.Dataset):
    def __init__(self, mode = 'train', augument = False):
        self.augument = augument
        self.mode = mode
        if self.mode == 'test':
            self.visit_root = '../data/npy/test_visit/'
            self.img_root = '../data/image/test/'
        else:
            self.visit_root = '../data/npy/train_visit/'
            self.img_root = '../data/image/train/'
        self.visit_root = self.img_root
        self.imgs = list()
        self.visit = list()

        dir = os.listdir(self.visit_root)
        for x, y in zip(os.listdir(self.visit_root), os.listdir(self.img_root)):
            self.visit.append(self.visit_root + x)
            self.imgs.append(self.img_root + y)
        self.imgs.sort()
        print(len(self.imgs))

    def __getitem__(self, item):
        if self.mode == 'dev':
            item += 380000
        img_id = self.imgs[item]
        X = cv2.imread(img_id)
        if self.augument:
            X = self.augumentor(X)
        img = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        visit = img
        #visit_id = self.visit[item]
        #visit = np.load(visit_id)
        if self.mode == 'test':
            return img, int(img_id[-9:-4])
        else:
            return img, int(img_id[-5]) - 1
        if self.mode == 'test':
            return img, torch.tensor(visit, dtype=torch.float32), int(visit_id[-9:-4])
        else:
            return img, torch.tensor(visit, dtype=torch.float32), int(visit_id[-5]) - 1

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug
    def __len__(self):
        if self.mode == 'train':
            return 380000
        elif self.mode == 'dev':
            return 20000
        else:
            return 100000
#reader = Reader()

