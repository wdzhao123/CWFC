from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from model import CNN, Bottleneck
from dataset import data_all
import warnings

warnings.filterwarnings("ignore")
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT_PATH)


def parse_args():
    # Test settings
    parser = argparse.ArgumentParser(description='Test script for CFCL')
    parser.add_argument('--model_path', type=str, default='save_model/MODEL_PATH',
                        help='model for testing')
    parser.add_argument('--data_dir', type=str, default='dataset/DATASET_DIR',
                        help='dataset for testing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    args = parser.parse_args()
    return args


def test_dir(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} as device." .format(device))
    model_path = os.path.join(ROOT_PATH, args.model_path)
    assert os.path.exists(model_path), "Model path does not exist."
    data_path = os.path.join(ROOT_PATH, args.data_dir)
    assert os.path.exists(os.path.join(data_path, 'test')), "Data path does not exist"

    testset = data_all(root=data_path, is_train=False, data_len=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers, drop_last=False)
    num_cls = testset.get_cls_num()
    model = CNN(Bottleneck, [3, 4, 6, 3], num_classes=num_cls).to(device)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    # 加载参数进行测试
    model.load_state_dict(torch.load(model_path))
    test_acc_ave = 0.0
    test_num = 0
    model.eval()
    num = np.zeros(num_cls, dtype=int)
    test_acc = np.zeros(num_cls, dtype=int)
    cs = np.zeros(num_cls, dtype=float)
    print('Begin testing...')
    for i, (images, labels) in enumerate(testloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, _, _, _, _ = model(images)
        _, prediction = torch.max(outputs.data, 1)

        z = int(labels)
        num[z] += 1
        test_acc[z] += torch.sum(prediction == labels.data)
        test_num = test_num + len(images)
        test_acc_ave = test_acc_ave + torch.sum(prediction == labels.data)
    print('Finish testing.')
    for i in range(num_cls):
        cs[i] = test_acc[i] / num[i]
        print(str(i) + ":" + str(cs[i]))
    test_acc_ave = test_acc_ave / test_num
    print("Overall accuracy" + ":" + str(test_acc_ave))
    return test_acc_ave


if __name__ == "__main__":
    args = parse_args()
    test_dir(args)


