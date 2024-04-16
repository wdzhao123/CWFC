import imageio
import numpy as np
import os
import PIL.Image as Image
from torchvision import transforms


class data_all():
    def __init__(self, root, threshold=0, is_train=True, data_len=None):
        self.root = root
        self.is_train = 'train' if is_train else 'test'
        self.t = threshold
        ann_path = os.path.join(self.root, 'anno', '{}_all.txt'.format(self.is_train))
        if not os.path.exists(ann_path):
            d_set = os.path.join(self.root, self.is_train)
            assert os.path.exists(d_set), "All {} data should be placed in the same directory".format(self.is_train)
            with open(ann_path, 'w') as ann:
                for cls in os.listdir(d_set):
                    dir = os.path.join(d_set, cls)
                    images_filename = os.listdir(dir)
                    for img in images_filename:
                        p = os.path.join(cls, img)
                        ann.writelines(p + ' ' + cls + '\n')
            ann.close()
        file_txt = open(ann_path)
        file_list = []
        label_list = []
        for line in file_txt:
            file_list.append(line[:-1].split(' ')[0])
            label_list.append(int(line[:-1].split(' ')[1]))

            self.img = [imageio.imread(os.path.join(self.root, self.is_train, file)) for file in
                             file_list[:data_len]]
            self.label = label_list[:data_len]

    def get_cls_num(self):
        d_set = os.path.join(self.root, self.is_train)
        cls = os.listdir(d_set)
        return len(cls)

    def get_subset(self):
        head_cls = []
        tail_cls = []
        train_set = os.path.join(self.root, 'train')
        for cls in os.listdir(train_set):
            dir = os.path.join(train_set, cls)
            s_num = len(os.listdir(dir))
            if s_num >= self.t:
                head_cls.append(int(cls))
            else:
                tail_cls.append(int(cls))
        return head_cls, tail_cls

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Grayscale(num_output_channels=1)(img)
        img = transforms.Resize((224, 224), Image.BICUBIC)(img)
        if self.is_train == 'train':
            img = transforms.RandomHorizontalFlip(p=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([.5], [.5])(img)

        return img, target

    def __len__(self):
        return len(self.label)


class data_part():
    def __init__(self, root, split, threshold=0, is_train=True, data_len=None):
        self.root = root
        self.is_train = 'train' if is_train else 'test'
        self.split = split
        self.t = threshold
        ann_path = os.path.join(self.root, 'anno/{}_{}.txt'.format(self.is_train, self.split))
        if not os.path.exists(ann_path):
            # Data partition (if not manually done)
            head_cls = []
            tail_cls = []
            train_set = os.path.join(self.root, 'train')
            for cls in os.listdir(train_set):
                dir = os.path.join(train_set, cls)
                s_num = len(os.listdir(dir))
                if s_num >= self.t:
                    head_cls.append(cls)
                else:
                    tail_cls.append(cls)
            dset = os.path.join(self.root, self.is_train)
            with open(ann_path, 'w') as ann:
                for cls in head_cls if split == 'head' else tail_cls:
                    dir = os.path.join(dset, cls)
                    images_filename = os.listdir(dir)
                    for img in images_filename:
                        p = os.path.join(cls, img)
                        ann.writelines(p + ' ' + cls + '\n')
            ann.close()
        file_txt = open(ann_path)
        file_list = []
        label_list = []
        for line in file_txt:
            file_list.append(line[:-1].split(' ')[0])
            label_list.append(int(line[:-1].split(' ')[1]))

            self.img = [imageio.imread(os.path.join(self.root, self.is_train, file)) for file in
                             file_list[:data_len]]
            self.label = label_list[:data_len]

    def get_cls_num(self):
        dset = os.path.join(self.root, self.is_train)
        cls = os.listdir(dset)
        return len(cls)

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Grayscale(num_output_channels=1)(img)
        img = transforms.Resize((224, 224), Image.BICUBIC)(img)
        if self.is_train == 'train':
            img = transforms.RandomHorizontalFlip(p=0.5)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([.5], [.5])(img)

        return img, target

    def __len__(self):
        return len(self.label)