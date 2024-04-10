# -*- coding:utf-8 -*-
import sys

from PIL import Image
import numpy as np
import math
import os
import glob
from tqdm import tqdm

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label):
    num_class = 6
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def Eval():
    num_class = 6
    IMAGE_FORMAT = '.png'
    
    INFER_DIR1 = './ablation_wo_SSFA/result/pred1/'  # Inference path
    INFER_DIR2 = './ablation_wo_SSFA/result/pred2/'  # Inference path
    LABEL_DIR1 = '../dataset/HRSCD_change_only/test/labels_2006/'  # GroundTruth path
    LABEL_DIR2 = '../dataset/HRSCD_change_only/test/labels_2012/'  # GroundTruth path

    infer_list1 = glob.glob(INFER_DIR1 + "*{}".format(IMAGE_FORMAT))
    infer_list2 = glob.glob(INFER_DIR2 + "*{}".format(IMAGE_FORMAT))

    infer_list1.sort()
    infer_list2.sort()

    infer_list = infer_list1 + infer_list2

    label_list1 = glob.glob(LABEL_DIR1 + "*{}".format(IMAGE_FORMAT))
    label_list2 = glob.glob(LABEL_DIR2 + "*{}".format(IMAGE_FORMAT))

    label_list1.sort()
    label_list2.sort()

    label_list = label_list1 + label_list2
    # print(label_list)
    # print(infer_list)
    assert len(label_list) == len(infer_list), "Predictions do not match targets length"
    assert set([os.path.basename(label) for label in label_list1]) == set([os.path.basename(infer) for infer in infer_list1]), "Predictions do not match targets name"
    assert set([os.path.basename(label) for label in label_list2]) == set([os.path.basename(infer) for infer in infer_list2]), "Predictions do not match targets name"

    hist = np.zeros((num_class, num_class))
    for infer, gt in tqdm(zip(infer_list, label_list)):
        try:
            infer = Image.open(infer)
        except:
            print("File open error")
            sys.exit(0)
        try:
            label = Image.open(gt)
        except:
            print("File open error")
            sys.exit(0)
        infer_array = np.array(infer)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"

        hist += get_hist(infer_array, label_array)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    #print(hist)
    #print(kappa_n0)
    print('Mean IoU = %.5f' % IoU_mean)
    print('Kappa = %.5f' % kappa_n0)
    return IoU_mean, kappa_n0


if __name__ == '__main__':
    num_class = 6
    Eval()

