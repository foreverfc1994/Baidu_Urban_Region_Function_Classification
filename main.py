import os
os.environ['CUDA_VISIBLE_DEVICES']='3'#3
import torch
import torch.nn as nn
import torch.utils.data as data
import argparse
import cv2
import torch.optim as optim
import time
import numpy as np
from torchvision import transforms as T
from imgaug import augmenters as iaa

layer = False#True 使用分层学习率

class Reader(data.Dataset):
    def __init__(self, mode = 'train', augument=False):
        self.augument = augument
        self.mode = mode
        if self.mode == 'test':
            self.img_root = '../data/image/test/'
            self.visit_root = '../data/npy/test_visit/'
        else:
            self.img_root = '../data/image/train/'
            self.visit_root = '../data/npy/train_visit/'
        self.visit = list()
        self.img = list()
        for x, y in zip(os.listdir(self.img_root), os.listdir(self.visit_root)):
            self.img.append(self.img_root + x)
            self.visit.append(self.visit_root + y)
        self.visit.sort()#存放文本数据所有的路径
        self.img.sort()

    def __getitem__(self, item):
        if self.mode == 'dev':
            item += 380000
        visit_id = self.visit[item]
        img_id = self.img[item]
        visit = np.load(visit_id)

        X = cv2.imread(img_id)
        if self.augument:
            X = self.augumentor(X)
        img = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        visit = torch.tensor(visit, dtype = torch.float32)
        visit = torch.log(visit + 1)
        #img.fill_(0)
        #visit.fill_(0)
        #visit = torch.cat((visit[3:-1], visit[0:3]), dim = 0)
        if self.mode == 'test':
            return img, visit, int(visit_id[-9:-4])
        else :
            return img, visit, int(visit_id[-5]) - 1
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
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
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

def adjust_learning_rate(optimizer, epoch, iteration, epoch_size, LR, lr_list):
    if epoch < 2:
        lr = 1e-6 + (LR-1e-6) * iteration / (epoch_size * 1)
    else:
        lr = lr_list[epoch - 1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if layer:
        optimizer.param_groups[0]['lr'] *= 3
    return lr

def more_feature(visit, in_channel = 14, dim = 4, type = 'sum'):
    if in_channel == 7:
        return visit
    elif in_channel == 14:
        if type == 'sum':
            s = visit.sum(dim=dim).unsqueeze(dim)#b, 1,7,26,24
        elif type == 'max':
            s, _ = visit.max(dim)
            s = s.unsqueeze(dim)
        buf = visit / (s + 1e-6)
        visit = torch.cat((visit, buf), dim=1)
        return visit

dim = 4 #dim=2,3,4分别训练
in_channel = 14 #in_channel =7训练一次，in_channel=14训练3次（dim=2，3，4），四个模型做集成
type = 'sum'
use_3D = True
use_more = False
#from sp_model import build_net
from f_model import build_net
def train():
    LR = 0.001
    epochs = 13
    batch_size = 64
    criterion = nn.CrossEntropyLoss()

    start = 0
    if start == 0:
        resume = None
    else:
        resume = './weights/' + str(start) + '.pth'

    net = build_net(in_channels=in_channel)
    net = net.cuda()
    if resume:
        net.load_state_dict(torch.load(resume))

    net = net.cuda()
    net.train()
    lr_list = []

    if layer:
        image = list(map(id,  net.img_encoder.parameters()))
        other_parameters = (p for p in net.parameters() if id(p) not in image)
        parameters = [{'params': net.img_encoder.parameters(), 'lr': LR * 3},
                      {'params': other_parameters, 'lr': LR}]
        optimizer = torch.optim.SGD(parameters, lr=LR, momentum=0.9, weight_decay=5e-4)
    else:
        for param in net.img_encoder.parameters():
            param.requires_grad = False
        image = list(map(id,  net.img_encoder.parameters()))
        other_parameters = (p for p in net.parameters() if id(p) not in image)
        optimizer = optim.SGD(other_parameters, lr=LR, momentum=0.9, weight_decay=5e-4)
        #optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    for epoch in range(epochs):
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    lr_list[6] /= 5
    lr_list[7] /= 20
    reader = Reader(augument=True)
    epoch_size = len(reader) // batch_size
    max_iter = epoch_size * epochs
    epoch = start
    min_iter = epoch * epoch_size
    for it in range(min_iter, max_iter):
        time1 = time.time()
        if it % epoch_size == 0:
            batch = iter(data.DataLoader(dataset=reader, batch_size=batch_size, num_workers=4, shuffle=True))
            if it != 0 and epoch % 1 == 0 and it != min_iter:
                torch.save(net.state_dict(), './weights/' + str(epoch) + '.pth')
            epoch += 1
        #if it % 500 == 0:
        #    torch.save(net.state_dict(), './weights/' + str(it) + '.pth')
        if epoch > epochs or epoch > 8:
            break
        lr = adjust_learning_rate(optimizer, epoch, it, epoch_size, LR=LR, lr_list=lr_list)
        img, visit, target = next(batch)
        img, visit, target = img.cuda(), visit.cuda(), target.cuda()
        if use_3D:
            visit = visit.unsqueeze(1)
        visit = more_feature(visit, in_channel=in_channel, dim=dim, type=type)

        out = net((img, visit))
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time2 = time.time()
        if it % 20 == 0:
            print(epoch, it, it % epoch_size, epoch_size, loss.item(), time2 - time1)
    torch.save(net.state_dict(), './weights/' + 'Final' + '.pth')

def dev():
    epochs = 1
    batch_size = 50
    net = build_net(in_channels=in_channel)
    net.eval()
    net.load_state_dict(torch.load('./weights/8.pth'))
    net = net.cuda()
    reader = Reader(mode='dev')
    epoch_size = len(reader) // batch_size
    max_iter = epoch_size * epochs
    batch = iter(data.DataLoader(dataset=reader, batch_size=batch_size, num_workers=4, shuffle=False))
    tot = 0
    sum = 0
    for it in range(max_iter):
        img, visit, target = next(batch)
        img, visit, target = img.cuda(), visit.cuda(), target.cuda()
        if use_3D:
            visit = visit.unsqueeze(1)
        visit = more_feature(visit, in_channel=in_channel, dim=dim, type=type)
        #visit.fill_(0)
        out = net((img, visit))
        _, predict = out.max(dim = 1)
        acc = predict == target
        acc = acc.sum()
        tot += acc
        sum += 50
        if it % 10 == 0:
            print(tot, sum)
    print(tot)

def infer():
    epochs = 1
    batch_size = 50

    net = build_net(in_channels=in_channel)
    net.load_state_dict(torch.load('./weights/12.pth'))#12
    net = net.cuda()
    net.eval()

    reader = Reader(mode='test')
    epoch_size = len(reader) // batch_size
    max_iter = epoch_size * epochs
    batch = iter(data.DataLoader(dataset=reader, batch_size=batch_size, num_workers=4, shuffle=False))
    with open("result.csv", 'w', encoding='utf-8') as f:
        for it in range(max_iter):
            img, visit, id = next(batch)
            img, visit, id = img.cuda(), visit.cuda(), id.cuda()
            if use_3D:
                visit = visit.unsqueeze(1)
            visit = more_feature(visit, in_channel=in_channel, dim=dim, type=type)
            out = net((img, visit))
            _, predict = out.max(dim = 1)
            id, predict = id.cpu().numpy(), predict.cpu().numpy()
            for i in range(batch_size):
                f.write("%06d"%id[i] + '\t' + "%03d"%(1 + predict[i]) + '\n')
            print(it)
    writecvs()
def new_infer():
    epochs = 1
    batch_size = 50

    net = build_net(in_channels=in_channel)
    net.load_state_dict(torch.load('./weights/8.pth'))
    net = net.cuda()
    net.eval()

    reader = Reader(mode='test')
    epoch_size = len(reader) // batch_size
    max_iter = epoch_size * epochs
    batch = iter(data.DataLoader(dataset=reader, batch_size=batch_size, num_workers=4, shuffle=False))
    with open("result.csv", 'w', encoding='utf-8') as f:
        for it in range(max_iter):
            if it % 10 == 0:
                print(it)
            img, visit, id = next(batch)
            img, visit, id = img.cuda(), visit.cuda(), id.cuda()
            if use_3D:
                visit = visit.unsqueeze(1)
            visit = more_feature(visit, in_channel=in_channel, dim=dim, type=type)
            out = net((img, visit))
            for x in out:
                x = x.cpu().detach().numpy()
                for j in range(9):
                    f.write('%f\t'%x[j])
                f.write('\n')
def writecvs():
    l = list()
    with open('result.csv', 'r', encoding='utf-8') as f1:
        for s in f1.readlines():
            ss = s[0:-1].split('\t')
            l.append((int(ss[0]), int(ss[1])))
    l.sort(key=lambda x: x[0])

    with open("result.csv", 'w', encoding='utf-8') as f2:
        for it in l:
            f2.write("%06d" % it[0] + '\t' + "%03d" % (it[1]) + '\n')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('mode',default=1,required=True,type=int)
    args.parse_args()
    if args.mode == 1:
        train()
    elif args.mode == 2:
        dev()
    elif args.mode == 3:
        infer()
    elif args.mode == 4:
        new_infer()

