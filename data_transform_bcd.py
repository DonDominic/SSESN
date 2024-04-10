import torch
import random
import numpy as np
from PIL import Image


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, image2, mask0_1):
        img1, img2, mask_bin = Image.fromarray(np.asarray(image)), Image.fromarray(np.asarray(image2)), Image.fromarray(np.asarray(mask0_1))

        # print("RandomHorizontalFlip image:", image)
        # print("RandomHorizontalFlip image2:", image2)
        rand = random.random()
        if rand < 1 / 6:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask_bin = mask_bin.transpose(Image.FLIP_LEFT_RIGHT)

        elif rand < 2 / 6:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask_bin = mask_bin.transpose(Image.FLIP_TOP_BOTTOM)

        elif rand < 3 / 6:
            img1 = img1.transpose(Image.ROTATE_90)
            img2 = img2.transpose(Image.ROTATE_90)
            mask_bin = mask_bin.transpose(Image.ROTATE_90)

        elif rand < 4 / 6:
            img1 = img1.transpose(Image.ROTATE_180)
            img2 = img2.transpose(Image.ROTATE_180)
            mask_bin = mask_bin.transpose(Image.ROTATE_180)

        elif rand < 5 / 6:
            img1 = img1.transpose(Image.ROTATE_270)
            img2 = img2.transpose(Image.ROTATE_270)
            mask_bin = mask_bin.transpose(Image.ROTATE_270)
        
        results = [img1, img2, mask_bin]
        # print("RandomHorizontalFlip image output:", image)
        # print("RandomHorizontalFlip image2 output:", image2)
        return results


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, *data):
        # pic,pic2,label=None,label2=None
        pic = data[0]
        pic2 = data[1]
        mask0_1= data[2]
        # print("pic:", pic)
        # print("pic2:", pic2)
        if isinstance(pic, np.ndarray) and isinstance(pic2, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
            img2 = torch.from_numpy(pic2)
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
            img2 = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic2.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            # print("img1:", img.size(), pic.size[1], pic.size[0])
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # print("img1:", img.size(), pic.size[1], pic.size[0])
            # print("img2:",img2.size(), pic2.size[1], pic2.size[0])
            img2 = img2.view(pic2.size[1], pic2.size[0], nchannel)
            # print("img2:", img2.size(), pic2.size[1], pic2.size[0])
            # put it from HWC to CHW format
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            img2 = img2.transpose(0, 1).transpose(0, 2).contiguous()
        # img = img.float().div(255)
        # img2 = img2.float().div(255)

        return img, img2, torch.LongTensor(np.array(mask0_1, dtype=np.int))


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
