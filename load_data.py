from torch.utils.data import Dataset
import torch
import numpy as np
import os

from tools import img_load, label_load


class IVData(Dataset):
    def __init__(self, data_folder, img_transform=None):
        self.img_folder = os.path.join(data_folder, "imgs")
        self.label_folder = os.path.join(data_folder, "labels")
        self.img_transform = img_transform
        self.images = os.listdir(self.img_folder)
        self.labels = os.listdir(self.label_folder)

    def __getitem__(self, index):
        #get img
        img_path = os.path.join(self.img_folder, self.images[index])
        img = img_load(img_path)
        if self.img_transform:
            img = self.img_transform(img)
        #get label
        label_path = os.path.join(self.label_folder, self.labels[index])
        label = label_load(label_path)
        label = torch.from_numpy(np.array(label).astype(int))
        return img, label
    
    def __len__(self):
        return len(self.images)



class IVData_F(Dataset):
    def __init__(self, data_folder, img_transform=None):
        self.data_folder = data_folder
        self.img_transform = img_transform
        self.labels = sorted(os.listdir(data_folder))
        self.sizes = []
        for label in self.labels:
            imgs = os.listdir(os.path.join(data_folder, label))
            self.sizes.append(len(imgs))
        print("labels: ", self.labels)
        print("sizes: ", self.sizes)

    def __getitem__(self, index):
        label = 0
        for size in self.sizes:
            if index < size:
                break
            index -= size
            label += 1
        label_path = os.path.join(self.data_folder, self.labels[label])
        img_name = os.listdir(label_path)[index]
        img_path = os.path.join(label_path, img_name)
        img = img_load(img_path)
        if self.img_transform:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return sum(self.sizes)


class IVData_D(Dataset):
    def __init__(self, data, img_transform=None):
        self.imgs = data[:,0]
        self.labels = data[:,1]
        self.img_transform = img_transform
        #print(self.imgs.shape)
        #print(self.labels.shape)
        #print(self.imgs)
        #print(self.labels)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = img_load(path)
        label = int(self.labels[index])
        if self.img_transform:
            img = self.img_transform(img)
        return img, label, path

    def __len__(self):
        return len(self.imgs)
        


class UnityData(Dataset):
    def __init__(self, data_folder, img_transform=None):
        self.data_folder = data_folder
        self.img_transform = img_transform
        self.labels = sorted(os.listdir(data_folder))
        self.sizes = []
        for label in self.labels:
            label = os.path.join(label, "Images")
            imgs = os.listdir(os.path.join(data_folder, label))
            self.sizes.append(len(imgs))
        #print("labels: ", self.labels)
        #print("sizes: ", self.sizes)

    def __getitem__(self, index):
        label = 0
        for size in self.sizes:
            if index < size:
                break
            index -= size
            label += 1
        label_path = os.path.join(self.data_folder, self.labels[label])
        label_path = os.path.join(label_path, "Images")
        img_name = os.listdir(label_path)[index]
        img_path = os.path.join(label_path, img_name)
        img = img_load(img_path)
        if self.img_transform:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return sum(self.sizes)


        