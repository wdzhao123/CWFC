from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
import argparse
import numpy as np
from model import CNN, Bottleneck, resnet50
from dataset import data_all, data_part
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT_PATH)


def parse_args():
    # Test settings
    parser = argparse.ArgumentParser(description='Student training script for CFCL')
    parser.add_argument('--head_teacher_path', type=str, default='save_model/HEAD_TEACHER_PATH',
                        help='teacher model of head subset')
    parser.add_argument('--tail_teacher_path', type=str, default='save_model/TAIL_TEACHER_PATH',
                        help='teacher model of tail subset')
    parser.add_argument('--model_save_dir', type=str, default='save_model/MODEL_SAVE_DIR',
                        help='directory to save trained models')
    parser.add_argument('--data_dir', type=str, default='dataset/DATASET_NAME',
                        help='dataset for training')
    parser.add_argument('--epoch', type=int, default=100,
                        help='training epoch')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--threshold', type=int, default=100,
                        help='threshold to divide head and tail subset')
    args = parser.parse_args()
    return args


def adjust_learning_rate(epoch, lr):
    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10
    return lr


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def get_center(args, split):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    data_path = os.path.join(ROOT_PATH, args.data_dir)
    assert os.path.exists(data_path), "Data path does not exist"
    trainset = data_part(root=data_path, split=split, threshold=args.threshold, is_train=True, data_len=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                  shuffle=True, num_workers=args.num_workers, drop_last=False)
    num_cls = trainset.get_cls_num()
    teacher = CNN(Bottleneck, [3, 4, 6, 3], num_classes=num_cls).to(device)
    teacher = nn.DataParallel(teacher).cuda()
    if split == 'head':
        teacher.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.head_teacher_path)))
    else:
        teacher.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.tail_teacher_path)))

    teacher.eval()
    center_feature = np.zeros((num_cls, 2048, 1, 1))
    num = np.zeros(num_cls)

    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        output, feature, _, _, _ = teacher(images)
        feature = feature.cpu().detach().numpy()
        _, prediction = torch.max(output.data, 1)
        if labels == prediction:
            center_feature[labels] += feature[0]
            num[labels] += 1

    for index in range(num_cls):
        if num[index] != 0:
            center_feature[index] = center_feature[index] / num[index]
    return center_feature


def train_s(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} as device.".format(device))

    # get center featuers
    center_feature = get_center(args, 'head') + get_center(args, 'tail')

    torch.cuda.empty_cache()
    save_path = os.path.join(ROOT_PATH, args.model_save_dir)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    data_path = os.path.join(ROOT_PATH, args.data_dir)
    assert os.path.exists(data_path), "Data path does not exist."

    trainset = data_all(root=data_path, threshold=args.threshold, is_train=True, data_len=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    testset = data_all(root=data_path, threshold=args.threshold, is_train=False, data_len=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=args.num_workers, drop_last=False)
    num_cls = trainset.get_cls_num()
    h_cls, t_cls = trainset.get_subset()
    teacher_h = CNN(Bottleneck, [3, 4, 6, 3], num_classes=num_cls).to(device)
    teacher_h = nn.DataParallel(teacher_h).cuda()
    teacher_h.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.head_teacher_path)))
    teacher_t = CNN(Bottleneck, [3, 4, 6, 3], num_classes=num_cls).to(device)
    teacher_t = nn.DataParallel(teacher_t).cuda()
    teacher_t.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.tail_teacher_path)))
    stu = CNN(Bottleneck, [3, 4, 6, 3], num_classes=num_cls).to(device)
    pretrained = resnet50(pretrained=True)
    pretrained_dict = pretrained.state_dict()
    model_dict = stu.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    stu.load_state_dict(model_dict)
    stu = nn.DataParallel(stu).cuda()

    optimizer = torch.optim.SGD(stu.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss(size_average=True, reduce=True)

    best_acc = 0.0
    best_epoch = 0
    print('Begin training...')
    for epoch in range(args.epoch):
        # train
        stu.train()
        train_acc = 0.0
        train_loss = 0.0
        img_all_train_num = 0
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)  # leave=False 进度在一行显示
        for i, (images, labels) in loop:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            requires_grad(teacher_t, False)
            requires_grad(teacher_h, False)
            requires_grad(stu, True)

            outputs, features, _, _, fea_3 = stu(images)
            _, prediction = torch.max(outputs.data, 1)

            cur_center_fea = np.zeros((len(images), 2048, 1, 1))
            for index in range(len(images)):
                truth = labels[index]
                cur_center_fea[index] = center_feature[truth]
            cur_center_fea = torch.tensor(cur_center_fea).to(device)
            features = features.flatten()
            cur_center_fea = cur_center_fea.flatten()

            fea_target = [0] * len(images)
            _, _, _, _, fea_h = teacher_h(images)
            fea_h = fea_h.cpu().detach().numpy()
            fea_h = fea_h.tolist()
            _, _, _, _, fea_t = teacher_t(images)
            fea_t = fea_t.cpu().detach().numpy()
            fea_t = fea_t.tolist()

            for k in range(len(labels)):
                if int(labels[k]) in h_cls:
                    fea_target[k] = fea_h[k]
                elif int(labels[k]) in t_cls:
                    fea_target[k] = fea_t[k]
            fea_target = torch.Tensor(fea_target).to(device)

            loss = loss_fn(outputs, labels)
            loss1 = loss_mse(features.float(), cur_center_fea.float())
            loss2 = loss_mse(fea_3, fea_target)
            loss_all = loss1 + loss + loss2
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            train_acc = train_acc + torch.sum(prediction == labels.data)
            train_loss += loss.item() * images.size(0)
            img_all_train_num = img_all_train_num + len(images)

            loop.set_description(f'Epoch [{epoch}/{args.epoch}]')

        lr = adjust_learning_rate(epoch, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        train_acc = train_acc/img_all_train_num
        train_loss = train_loss/img_all_train_num

        # eval
        test_acc = 0.0
        num = 0
        stu.eval()
        for i, (images, labels) in enumerate(testloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            outputs, _, _, _, _ = stu(images)
            _, prediction = torch.max(outputs.data, 1)
            num = num + len(images)
            test_acc = test_acc + torch.sum(prediction == labels.data)
        test_acc = test_acc / num
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        model_save_path = os.path.join(save_path, "epoch_{}".format(epoch))
        torch.save(stu.state_dict(), model_save_path)
        print("Finish epoch {}, Train_loss: {}, Train_acc: {}, Test_acc: {}.".format(epoch, train_loss, train_acc, test_acc))
    print('Finish student model training, highest accuracy: {} @ epoch_{}.'.format(best_acc, best_epoch))


if __name__ == "__main__":
    args = parse_args()
    # Train student models
    train_s(args)


