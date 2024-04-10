import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sscdnet import DeepLabv3_plus
from dataset_SECOND import PCD, PCD_full

checkpointdir = './ablation_wo_CA'
datadir = '../dataset/HRSCD_change_only'

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def colormap():
    cmap = np.zeros([6, 3]).astype(np.uint8)
    cmap[0, :] = np.array([255, 255, 255])
    cmap[1, :] = np.array([128, 0, 0])  # Artificial surfaces, deep red
    cmap[2, :] = np.array([128, 128, 128])  # Agricultural areas, gray
    cmap[3, :] = np.array([0, 128, 0])  # Forests, deep green
    cmap[4, :] = np.array([0, 255, 0])  # Wetlands, shallow green
    cmap[5, :] = np.array([0, 0, 255])  # Water, blue

    return cmap

class Colorization:

    def __init__(self, n=6):
        self.cmap = colormap()
        self.cmap = torch.from_numpy(np.array(self.cmap[:n]))

    def __call__(self, gray_image):
        color_image = torch.ByteTensor(3, 512, 512).fill_(255)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][2]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][0]

        return color_image

color_transform = Colorization(6)

def visulization():
    ckptname = 'best.pth'
    model = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=False)
    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load(os.path.join(checkpointdir, ckptname)))
    model.eval()
    loader_test = DataLoader(
        PCD_full(os.path.join(datadir, 'test')),
        num_workers=1, batch_size=1, shuffle=False)
    tqdm_test = tqdm(loader_test)

    for t0, t1, filename in tqdm_test:
        inputs = torch.from_numpy(np.concatenate((t0[0], t1[0]), axis=0)).unsqueeze(0).cuda()
        # Get predictions
        output0, output1, output_gray = model(inputs)

        output0 = F.softmax(output0, dim=1)
        mask_pred0 = output0[0].cpu().max(0)[1]
        output1 = F.softmax(output1, dim=1)
        mask_pred1 = output1[0].cpu().max(0)[1]
        change_pred = (output_gray[0].cpu().max(0)[1].data.numpy() * 255).astype(np.uint8)

        save_results(mask_pred0, mask_pred1, change_pred, filename[0])

def save_results(mask_pred_1, mask_pred_2, change_pred, file_name):
    # w = h = 512
    # fn_img_t0 = os.path.join(datadir, 'test/images_2006', file_name + '.png')
    # fn_img_t1 = os.path.join(datadir, 'test/images_2012', file_name + '.png')
    fn_label0 = os.path.join(datadir, 'test/labels_2006', file_name + '.png')
    fn_label1 = os.path.join(datadir, 'test/labels_2012', file_name + '.png')

    # t0 = cv2.imread(fn_img_t0, cv2.IMREAD_COLOR)
    # t1 = cv2.imread(fn_img_t1, cv2.IMREAD_COLOR)
    label0 = cv2.imread(fn_label0, cv2.IMREAD_UNCHANGED)/1.0
    label1 = cv2.imread(fn_label1, cv2.IMREAD_UNCHANGED)/1.0

    mask_pred_disp1 = np.transpose(color_transform(
        mask_pred_1[np.newaxis, :, :].data).numpy(), (1, 2, 0)).astype(np.uint8)
    mask_pred_disp2 = np.transpose(color_transform(
        mask_pred_2[np.newaxis, :, :].data).numpy(), (1, 2, 0)).astype(np.uint8)
    label_disp1 = np.transpose(color_transform(
        torch.from_numpy(label0[np.newaxis, :, :])).numpy(), (1, 2, 0)).astype(np.uint8)
    label_disp2 = np.transpose(color_transform(
        torch.from_numpy(label1[np.newaxis, :, :])).numpy(), (1, 2, 0)).astype(np.uint8)

    # img_out = np.zeros((h * 3, w * 2, 3), dtype=np.uint8)
    # img_out[0:h, 0:w, :] = t0
    # img_out[0:h, w:w * 2, :] = t1
    # img_out[h:h * 2, 0:w, :] = mask_pred_disp1
    # img_out[h:h * 2, w:w * 2, :] = mask_pred_disp2
    # img_out[h * 2:h * 3, 0:w, :] = label_disp1
    # img_out[h * 2:h * 3, w:w*2, :] = label_disp2

    # dn_save_fuck = os.path.join(checkpointdir, 'vis', 'fuck_local')
    # fn_save_fuck = os.path.join(dn_save_fuck, file_name + '.png')
    dn_save0 = os.path.join(checkpointdir, 'vis', 'change')
    fn_save0 = os.path.join(dn_save0, file_name + '.png')
    dn_save1 = os.path.join(checkpointdir, 'vis', 'label2006')
    fn_save1 = os.path.join(dn_save1, file_name + '.png')
    dn_save2 = os.path.join(checkpointdir, 'vis', 'label2012')
    fn_save2 = os.path.join(dn_save2, file_name + '.png')
    dn_save3 = os.path.join(checkpointdir, 'vis', 'pred2006')
    fn_save3 = os.path.join(dn_save3, file_name + '.png')
    dn_save4 = os.path.join(checkpointdir, 'vis', 'pred2012')
    fn_save4 = os.path.join(dn_save4, file_name + '.png')

    check_dir(dn_save0)
    check_dir(dn_save1)
    check_dir(dn_save2)
    check_dir(dn_save3)
    check_dir(dn_save4)
    # check_dir(dn_save_fuck)

    cv2.imwrite(fn_save0, change_pred)
    cv2.imwrite(fn_save1, label_disp1)
    cv2.imwrite(fn_save2, label_disp2)
    cv2.imwrite(fn_save3, mask_pred_disp1)
    cv2.imwrite(fn_save4, mask_pred_disp2)
    # cv2.imwrite(fn_save_fuck, img_out)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    visulization()
