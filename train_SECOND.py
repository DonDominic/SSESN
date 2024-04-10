# -*- coding: utf-8 -*-
from __future__ import print_function

import multiprocessing
from argparse import ArgumentParser
from tqdm import tqdm
import datetime
import cv2
import random
import os
import numpy as np

import torch
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset_SECOND import PCD, PCD_full
from sscdnet import DeepLabv3_plus
from utils.loss import SegmentationLosses
from SCDD_eval import Eval
import data_transforms as transform
from sklearn.metrics import confusion_matrix

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def colormap():
    cmap=np.zeros([7, 3]).astype(np.uint8)

    cmap[0,:] = np.array([255, 255, 255])   # unchanged, white
    cmap[1,:] = np.array([0, 0, 255])       # water, blue
    cmap[2,:] = np.array([128, 128, 128])   # ground, gray
    cmap[3,:] = np.array([0, 128, 0])       # low vegetation, deep green
    cmap[4,:] = np.array([0, 255, 0])       # tree, shallow green
    cmap[5,:] = np.array([128, 0, 0])       # building, deep red
    cmap[6,:] = np.array([255, 0, 0])       # playgrounds, shallow red

    return cmap

class Colorization:
    def __init__(self, n=7):
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

class Training:
    def __init__(self, arguments):
        self.args = arguments
        self.icount = 0
        self.dn_save = self.args.checkpointdir
        check_dir(self.dn_save)

        t = []
        t.extend([
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
        ])
        Com = transform.Compose(t)
        self.Com=Com


    def train(self, epoch):
        self.color_transform = Colorization(6)

        # Dataset loader for train
        dataset_train = DataLoader(
            PCD(os.path.join(self.args.datadir, 'train'), self.Com),
            num_workers=self.args.num_workers, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.test_path = self.dn_save
        check_dir(self.test_path)

        # Set loss function
        self.criterion = SegmentationLosses(
            weight=None, cuda=True).build_loss(
            mode=self.args.loss_type)

        # Resuming checkpoint
        self.best_pred = 0.0

        if self.args.resume is not None and self.flag==0:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'".format(
                        self.args.resume))
            checkpoint = torch.load(self.args.resume)
            #self.args.start_epoch = checkpoint['epoch']
            # epoch= checkpoint['epoch']
            if False:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not self.args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
            self.flag=1
        icount_loss = []
        color_loss = []
        gray_loss = []
        # tbar = tqdm(dataset_train)
        print('epoch {} ----------------------------------------'.format(epoch))
        # self.model_lr_scheduler.step(epoch)
        self.model.train()
        self.loss_log=[]
        self.loss_color_log=[]
        self.loss_gray_log=[]
        for step, (inputs_train, mask0_train, mask1_train, mask01_train) in enumerate(dataset_train):
            # Variables
            inputs_train = inputs_train.cuda()
            mask0_train = mask0_train.cuda()
            mask1_train = mask1_train.cuda()
            mask01_train = mask01_train.cuda()
            output0_train, output1_train, gray_outpout = self.model(inputs_train)

            # Optimizer
            self.optimizer.zero_grad()
            loss_color = self.criterion(output0_train, mask0_train) + self.criterion(output1_train, mask1_train)
            loss_gray = self.criterion(gray_outpout, mask01_train)
            loss = loss_color + loss_gray * 2
            loss.backward()
            self.optimizer.step()

            icount_loss.append(loss.item())
            color_loss.append(loss_color.item())
            gray_loss.append(loss_gray.item())
            if step % 21 == 0 and step is not 0:
                average = sum(icount_loss) / len(icount_loss)
                average_color = sum(color_loss) / len(color_loss)
                average_gray = sum(gray_loss) / len(gray_loss)
                self.loss_log.append(average)
                self.loss_color_log.append(average_color)
                self.loss_gray_log.append(average_gray)
                print('time: {}, loss: {:.6f}, loss_color: {:.6f}, loss_gray: {:.6f}, lr_backbone: {:.8f}, lr: {:.8f} (step: {})'.format(
                    datetime.datetime.now(), average, average_color, average_gray, self.optimizer.state_dict()['param_groups'][0]['lr'], self.optimizer.state_dict()['param_groups'][1]['lr'], step))
                icount_loss.clear()
                color_loss.clear()
                gray_loss.clear()

        self.checkpoint2(epoch)


    def checkpoint2(self, epoch):
        c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        filename = 'second-epoch{}.pth'.format(epoch)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.dn_save, filename))
        print('save: {0} (epoch: {1})'.format(filename, epoch))

        # self.model.load_state_dict(torch.load(os.path.join(self.dn_save, filename)))
        self.model.eval()
        loader_test = DataLoader(
            PCD_full(os.path.join(self.args.datadir, 'test')),
            num_workers=1, batch_size=1, shuffle=False)
        tqdm_test = tqdm(loader_test)
        for t0, t1, filename in tqdm_test:
            inputs = torch.from_numpy(np.concatenate((t0[0], t1[0]), axis=0)).unsqueeze(0).cuda()
            # Get predictions
            output0, output1, output_gray = self.model(inputs)

            output0 = F.softmax(output0, dim=1)
            mask_pred0 = output0[0].cpu().max(0)[1]
            output1 = F.softmax(output1, dim=1)
            mask_pred1 = output1[0].cpu().max(0)[1]
            
            fn_change = os.path.join(self.args.datadir, 'test/mask0_1', filename[0] + '.png')
            labels = cv2.imread(fn_change, cv2.IMREAD_UNCHANGED) / 1.0
            _, cd_preds = torch.max(output_gray, 1)
            tn, fp, fn, tp = confusion_matrix(labels.astype("int").flatten(),
                        cd_preds.data.cpu().numpy().flatten(), labels=[0,1]).ravel()

            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

            self.save_results(mask_pred0, mask_pred1, filename[0])
        tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F1 = 2 * P * R / (R + P)
        OA = (tp + tn) / (tp + tn + fp + fn)
        print('Precision: {}\nRecall: {}\nF1-Score: {}\nOA: {}\n'.format(P, R, F1, OA))
        mIoU, kappa = Eval()
        fn_score = os.path.join(self.args.checkpointdir, 'scores.txt')
        self.f_score = open(fn_score, 'a')
        self.f_score.write('epoch:{} --------------------------------{}\n'.format(epoch,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.f_score.write('Precision: {}\nRecall: {}\nF1-Score: {}\nOA: {}\n'.format(P, R, F1, OA))
        self.f_score.write('Kappa: {} \nmIoU: {}\n'.format(kappa, mIoU))
        self.f_score.close()

        if self.OA < OA or self.kappa < kappa:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.dn_save, 'best.pth'))
            self.OA = OA
            self.kappa = kappa


    def save_results(self, mask_pred_1, mask_pred_2, file_name):
        dn_save3 = os.path.join(self.args.checkpointdir, 'result', 'pred1')
        fn_save3 = os.path.join(dn_save3, file_name + '.png')
        dn_save4 = os.path.join(self.args.checkpointdir, 'result', 'pred2')
        fn_save4 = os.path.join(dn_save4, file_name + '.png')

        check_dir(dn_save3)
        check_dir(dn_save4)

        cv2.imwrite(fn_save3, mask_pred_1.data.numpy().astype(np.uint8))
        cv2.imwrite(fn_save4, mask_pred_2.data.numpy().astype(np.uint8))


    def run(self):
        print('Siamese Change Detection Network-----------------')
        # Define network
        self.model = DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True, _print=False)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.OA = self.kappa = 0.0
        self.iters = 0
        self.total_iters = 148 * self.args.epochs

        # Setting optimizer and learning rate
        self.optimizer = Adam([{"params": self.model.module.backbone.parameters(), "lr": self.args.lr * 0.1},
                    {"params": [param for name, param in self.model.module.named_parameters()
                                    if "backbone" not in name], "lr": self.args.lr}],
                    weight_decay=1e-4)

        # Resuming checkpoint
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'".format(
                        self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']

        self.flag = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if epoch == 5:
                lr = self.args.lr * 0.1
                self.optimizer.param_groups[0]["lr"] = lr * 0.1
                self.optimizer.param_groups[1]["lr"] = lr
            self.train(epoch)


if __name__ == '__main__':
    parser = ArgumentParser(description='SSESN scd Training')
    parser.add_argument('--checkpointdir', required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--icount-plot', type=int, default=100)
    parser.add_argument('--icount-save', type=int, default=5000)
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--seed', type=int, default=111111,
                        help='random seed')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--dataset', type=str, default='cd',
                        choices=['cd', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    seed_torch(args.seed)
    training = Training(args)
    training.run()
