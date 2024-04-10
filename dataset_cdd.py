import numpy as np
import os
import cv2
from numpy.core.fromnumeric import mean
import torch
from torch.utils.data import Dataset
EXTENSIONS = ['jpg','.png']

def check_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def get_img_path(root, basename, extension):
    return os.path.join(root, basename+extension)

def get_img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class CDD(Dataset):

    def __init__(self, root, Com):
        super(CDD, self).__init__()
        self.img_t0_root = os.path.join(root, 'A')
        self.img_t1_root = os.path.join(root, 'B')
        self.mask_01 = os.path.join(root, 'OUT')

        self.filenames = [get_img_basename(f) for f in os.listdir(self.img_t0_root) if check_img(f)]
        self.filenames.sort()
        self.Com = Com
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        print('{}:{}'.format(root,len(self.filenames)))

    def __getitem__(self, index):
        filename = self.filenames[index]

        fn_img_t0 = get_img_path(self.img_t0_root, filename, '.jpg')
        fn_img_t1 = get_img_path(self.img_t1_root, filename, '.jpg')
        fn_mask_01 = get_img_path(self.mask_01, filename, '.jpg')
        if os.path.isfile(fn_img_t0) == False:
            print ('Error: File Not Found: ' + fn_img_t0)
            exit(-1)
        if os.path.isfile(fn_img_t1) == False:
            print ('Error: File Not Found: ' + fn_img_t1)
            exit(-1)
        if os.path.isfile(fn_mask_01) == False:
            print ('Error: File Not Found: ' + fn_mask_01)
            exit(-1)

        img_t0 = cv2.imread(fn_img_t0, cv2.IMREAD_COLOR)
        img_t1 = cv2.imread(fn_img_t1, cv2.IMREAD_COLOR)
        mask0_1= cv2.imread(fn_mask_01 , cv2.IMREAD_GRAYSCALE)
        t0 = np.asarray(img_t0)
        t1 = np.asarray(img_t1)
        mask0_1 = np.asarray(mask0_1) / 255.

        data=[t0, t1, mask0_1]
        data = self.Com(*data)

        img_t0_ = (np.asarray(data[0]).astype("f") / 255.0 - self.mean) / self.std
        img_t1_ = (np.asarray(data[1]).astype("f") / 255.0 - self.mean) / self.std
        
        input_ = torch.from_numpy(np.concatenate((img_t0_[:, :, :], img_t1_[:, :, :]), axis=0))
        mask01 = torch.from_numpy(data[2].data.numpy()).long()

        return input_, mask01

    def __len__(self):
        return len(self.filenames)

    def get_random_index(self):
        index = np.random.randint(0, len(self.filenames))
        return index


class CDD_full(Dataset):

    def __init__(self, root):
        super(CDD_full, self).__init__()
        self.img_t0_root = os.path.join(root, 'A')
        self.img_t1_root = os.path.join(root, 'B')
        self.mask_01 = os.path.join(root, 'OUT')

        self.filenames = [get_img_basename(f) for f in os.listdir(self.img_t0_root) if check_img(f)]
        self.filenames.sort()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __getitem__(self, index):
        filename = self.filenames[index]

        fn_img_t0 = get_img_path(self.img_t0_root, filename, '.jpg')
        fn_img_t1 = get_img_path(self.img_t1_root, filename, '.jpg')
        fn_mask_01 = get_img_path(self.mask_01, filename, '.jpg')

        if os.path.isfile(fn_img_t0) == False:
            print ('Error: File Not Found: ' + fn_img_t0)
            exit(-1)
        if os.path.isfile(fn_img_t1) == False:
            print ('Error: File Not Found: ' + fn_img_t1)
            exit(-1)
        if os.path.isfile(fn_mask_01) == False:
            print ('Error: File Not Found: ' + fn_mask_01)
            exit(-1)

        img_t0 = cv2.imread(fn_img_t0, cv2.IMREAD_COLOR)
        img_t1 = cv2.imread(fn_img_t1, cv2.IMREAD_COLOR)
        mask0_1= cv2.imread(fn_mask_01 , cv2.IMREAD_GRAYSCALE)

        img_t0_ = (np.asarray(img_t0).astype("f").transpose(2, 0, 1) / 255.0 - self.mean) / self.std
        img_t1_ = (np.asarray(img_t1).astype("f").transpose(2, 0, 1) / 255.0 - self.mean) / self.std
        mask0_1 = torch.from_numpy((np.asarray(mask0_1) / 255.).astype("int")).long()
        
        return img_t0_, img_t1_, mask0_1, filename

    def __len__(self):
        return len(self.filenames)


