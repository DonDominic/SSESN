import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sscdnet import DeepLabv3_plus
from dataset_cdd import CDD, CDD_full

checkpointdir = './cdd_checkpoint'
datadir = '../dataset/CDD/subset'

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def visulization():
    ckptname = 'best.pth'
    model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=False, _print=False)
    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load(os.path.join(checkpointdir, ckptname)))
    model.eval()
    loader_test = DataLoader(
        CDD_full(os.path.join(datadir, 'test')),
        num_workers=1, batch_size=1, shuffle=False)
    tqdm_test = tqdm(loader_test)

    for t0, t1, labels, filename in tqdm_test:
        inputs = torch.from_numpy(np.concatenate((t0[0], t1[0]), axis=0)).unsqueeze(0).cuda()
        output = model(inputs)
        pred = (output[0].cpu().max(0)[1].unsqueeze(-1).data.numpy() * 255).astype(np.uint8)
        vis = display_results(pred, filename[0])
        save_results(pred, vis, filename[0])
        

def display_results(pred, filename):
    fn_img_t0 = os.path.join(datadir, 'test/A', filename + '.jpg')
    fn_img_t1 = os.path.join(datadir, 'test/B', filename + '.jpg')
    fn_mask = os.path.join(datadir, 'test/OUT', filename + '.jpg')

    t0 = cv2.imread(fn_img_t0, cv2.IMREAD_COLOR)
    t1 = cv2.imread(fn_img_t1, cv2.IMREAD_COLOR)
    gt = cv2.imread(fn_mask, cv2.IMREAD_GRAYSCALE)

    rows = cols = 256
    img_out = np.zeros((rows * 2, cols * 2, 3), dtype=np.uint8)
    img_out[0:rows, 0:cols, :] = t0
    img_out[0:rows, cols:cols * 2, :] = t1
    img_out[rows:rows * 2, cols:cols * 2, 2] = pred[:, :, 0]
    img_out[rows:rows * 2, 0:cols, 0] = gt

    return img_out


def save_results(pred, vis, filename):
    dn_save_pred = os.path.join(checkpointdir, 'pred_label')
    fn_save_pred = os.path.join(dn_save_pred, filename + '.jpg')
    check_dir(dn_save_pred)
    dn_save_vis = os.path.join(checkpointdir, 'vis')
    fn_save_vis = os.path.join(dn_save_vis, filename + '.jpg')
    check_dir(dn_save_vis)

    cv2.imwrite(fn_save_pred, pred)
    cv2.imwrite(fn_save_vis, vis)


if __name__ == '__main__':
    visulization()
