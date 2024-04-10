# -*- coding: utf-8 -*-
from __future__ import print_function

import multiprocessing
from argparse import ArgumentParser
from tqdm import tqdm
import datetime
import random
import os
import numpy as np

import torch
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset_cdd import CDD, CDD_full
from sscdnet_woCA_bcd import DeepLabv3_plus
from utils.loss import SegmentationLosses
import data_transform_bcd as transform
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
        os.mkdir(dir)


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
        # Dataset loader for train
        dataset_train = DataLoader(
            CDD(os.path.join(self.args.datadir, 'train'), self.Com),
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
        print('epoch {} ----------------------------------------'.format(epoch))
        # self.model_lr_scheduler.step(epoch)
        self.model.train()
        self.loss_log=[]
        for step, (inputs_train, mask01_train) in enumerate(dataset_train):
            # self.model.train()
            inputs_train = inputs_train.cuda()
            mask01_train = mask01_train.cuda()

            # print("in:",inputs_train.size()) [8, 6, 512, 512]
            gray_outpout = self.model(inputs_train)
            self.optimizer.zero_grad()
            # print(output0_train.size())torch.Size([4, 7, 512, 512])
            # print(mask0_train[:,0].size())torch.Size([4, 512, 512])
            loss = self.criterion(gray_outpout, mask01_train)
            loss.backward()
            self.optimizer.step()

            icount_loss.append(loss.item())
            if step % 25 == 0 and step is not 0:
                #self.test()
                average = sum(icount_loss) / len(icount_loss)
                self.loss_log.append(average)
                print('time: {}, loss: {:.6f}, lr_backbone: {:.8f}, lr: {:.8f} (step: {})'.format(
                    datetime.datetime.now(), average, self.optimizer.state_dict()['param_groups'][0]['lr'], self.optimizer.state_dict()['param_groups'][1]['lr'], step))
                icount_loss.clear()

        if epoch % 1 == 0 and epoch > 0:
            self.checkpoint2(epoch)

    def checkpoint2(self, epoch):
        c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        filename = 'cdd-epoch{}.pth'.format(epoch)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.dn_save, filename))
        print('save: {0} (epoch: {1})'.format(filename, epoch))

        # self.model.load_state_dict(torch.load(os.path.join(self.dn_save, filename)))
        self.model.eval()
        loader_test = DataLoader(
            CDD_full(os.path.join(self.args.datadir, 'test')),
            num_workers=1, batch_size=1, shuffle=False)
        tqdm_test = tqdm(loader_test)
        for t0, t1, labels, _ in tqdm_test:
            inputs = torch.from_numpy(np.concatenate((t0[0], t1[0]), axis=0)).unsqueeze(0).cuda()
            # Get predictions
            cd_preds = self.model(inputs)

            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten(), labels=[0,1]).ravel()

            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

        tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F1 = 2 * P * R / (R + P)
        fn_score = os.path.join(self.args.checkpointdir, 'scores.txt')
        self.f_score = open(fn_score, 'a')
        self.f_score.write('epoch:{} --------------------------------{}\n'.format(epoch,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.f_score.write('tn, fp, fn, tp = {}, {}, {}, {} \n'.format(tn, fp, fn, tp))
        self.f_score.write('Precision      = {} \n'.format(P))
        self.f_score.write('Recall         = {} \n'.format(R))
        self.f_score.write('F1             = {} \n\n'.format(F1))
        self.f_score.close()

        if self.P < P or self.R < R or self.F1 < F1:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.dn_save, 'best.pth'))
            self.P = P
            self.R = R
            self.F1 = F1

    def run(self):
        print('Siamese Change Detection Network-----------------')
        # Define network
        self.model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True, _print=False)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.P = self.R = self.F1 = 0.0
        self.iters = 0
        self.total_iters = 625 * self.args.epochs
        self.lr = self.args.lr

        # Setting optimizer and learning rate
        self.optimizer = Adam([{"params": self.model.module.backbone.parameters(), "lr": self.args.lr * 0.1},
                    {"params": [param for name, param in self.model.module.named_parameters()
                                    if "backbone" not in name], "lr": self.args.lr}],
                    weight_decay=1e-4)
        # self.model_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.5)

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
            if epoch == 50:
                lr = self.args.lr * 0.5
                self.optimizer.param_groups[0]["lr"] = lr * 0.1
                self.optimizer.param_groups[1]["lr"] = lr
            self.train(epoch)


if __name__ == '__main__':
    parser = ArgumentParser(description='SSESN bcd Training')
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
    parser.add_argument('--dataset', type=str, default='cd',
                        choices=['cd', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed_torch(args.seed)
    training = Training(args)
    training.run()
