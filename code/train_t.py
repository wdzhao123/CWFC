from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
import argparse
from model import CNN, Bottleneck, resnet50
from dataset import data_part
import warnings

warnings.filterwarnings("ignore")
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT_PATH)


def parse_args():
    # Test settings
    parser = argparse.ArgumentParser(description='Teacher training script for CFCL')
    parser.add_argument('--model_save_dir', type=str, default='save_model/MODEL_DIR',
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
        lr = lr/1000000
    elif epoch > 150:
        lr = lr/100000
    elif epoch > 120:
        lr = lr/10000
    elif epoch > 90:
        lr = lr/1000
    elif epoch > 60:
        lr = lr/100
    elif epoch > 30:
        lr = lr/10
    return lr


def train_t(args, split):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} as device." .format(device))
    torch.cuda.empty_cache()
    save_path = os.path.join(ROOT_PATH, args.model_save_dir, split)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    data_path = os.path.join(ROOT_PATH, args.data_dir)
    assert os.path.exists(data_path), "Data path does not exist."

    trainset = data_part(root=data_path, split=split, is_train=True, threshold=args.threshold, data_len=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    testset = data_part(root=data_path, split=split, is_train=False, threshold=args.threshold, data_len=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=args.num_workers, drop_last=False)
    num_cls = trainset.get_cls_num()
    model = CNN(Bottleneck, [3, 4, 6, 3], num_classes=num_cls).to(device)
    pretrained = resnet50(pretrained=True)
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()

    best_acc = 0.0
    best_epoch = 0
    print('Begin training...')
    for epoch in range(args.epoch):
        # train
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        num = 0
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs, _, _, _, _ = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)
            num = num + len(images)

        lr = adjust_learning_rate(epoch, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        train_acc = train_acc/num
        train_loss = train_loss/num

        # eval
        test_acc = 0.0
        num = 0
        model.eval()
        for i, (images, labels) in enumerate(testloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            outputs, _, _, _, _ = model(images)
            _, prediction = torch.max(outputs.data, 1)
            num = num + len(images)
            test_acc = test_acc + torch.sum(prediction == labels.data)
        test_acc = test_acc / num
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        model_save_path = os.path.join(save_path, "epoch_{}".format(epoch))
        torch.save(model.state_dict(), model_save_path)
        print("Finish epoch {}, Train_loss: {}, Train_acc: {}, Test_acc: {}.".format(epoch, train_loss, train_acc, test_acc))
    print('Finish {} teacher model training, highest accuracy: {} @ epoch_{}.'.format(split, best_acc, best_epoch))


if __name__ == "__main__":
    args = parse_args()
    # Train teacher models
    for split in ['head', 'tail']:
        train_t(args, split)


