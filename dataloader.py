'''Dataset loader of FER'''
from __future__ import print_function
from PIL import Image
import cv2
import numpy as np
import h5py
import torch.utils.data as data

class DataLoader(data.Dataset):
    def __init__(self, train_datasets, val_datasets, test_datasets, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set or val set
        self.train_data_h5 = h5py.File(train_datasets)
        self.val_data_h5 = h5py.File(val_datasets)
        self.test_data_h5 = h5py.File(test_datasets)
        ###rgb_19w_expand_64
        #self.train_data_h5 = h5py.File('/home/xuechao.shi/FER/code/Facial-Expression-Recognition.Pytorch-master/data/train_data_20200402/data_all_rgb_h5_72_expand/train_alldata_all_rgb_72_expand_0402.h5', 'r', driver='core')
        #self.train_data_h5 = h5py.File('/home/xuechao.shi/FER/code/Facial-Expression-Recognition.Pytorch-master/data/train_data_20200402/data_all_sample_rgb_h5_72_expand/train_alldata_all_sample_rgb_72_expand_0402.h5', 'r', driver='core')
        #self.train_data_h5 = h5py.File('/home/xuechao.shi/FER/code/Facial-Expression-Recognition.Pytorch-master/data/train_data_20200402/data_size_sample_rgb_h5_72_expand/train_alldata_size_sample_rgb_72_expand_0402.h5', 'r', driver='core')
        #self.val_data_h5 = h5py.File('/home/xuechao.shi/FER/code/Facial-Expression-Recognition.Pytorch-master/data/affect_rafdb_data/data_rgb_h5_72_20w_expand/val_raf_expand_label_rgb_72.h5', 'r', driver='core')
        #self.test_data_h5 = h5py.File('/home/xuechao.shi/FER/code/Facial-Expression-Recognition.Pytorch-master/data/affect_rafdb_data/data_rgb_h5_72_20w_expand/test_affect_expand_label_rgb_72.h5', 'r', driver='core')
        ###rgb_19w_expand_128
        # #self.train_data_h5 = h5py.File('./data/train_data_20200402/data_all_rgb_h5_144_expand/train_alldata_all_rgb_144_expand_0402.h5', 'r', driver='core')
        # self.train_data_h5 = h5py.File('./data/train_data_20200402/data_all_sample_rgb_h5_144_expand/train_alldata_all_sample_rgb_144_expand_0402.h5', 'r', driver='core')
        # #self.train_data_h5 = h5py.File('./data/train_data_20200402/data_size_sample_rgb_h5_144_expand/train_alldata_size_sample_rgb_144_expand_0402.h5', 'r', driver='core')
        # #self.train_data_h5 = h5py.File('./data/affect_rafdb_data/data_rgb_h5_144_20w_expand/train_affect_raf_label_rgb_144_20w_expand.h5', 'r', driver='core')
        # self.val_data_h5 = h5py.File('./data/affect_rafdb_data/data_rgb_h5_144_20w_expand/val_raf_expand_label_rgb_144.h5', 'r', driver='core')
        # self.test_data_h5 = h5py.File('./data/affect_rafdb_data/data_rgb_h5_144_20w_expand/test_affect_expand_label_rgb_144.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            number_class = len(self.train_data_h5['Train_label'])
            self.train_data = self.train_data_h5['Train_pixel']
            self.train_labels = self.train_data_h5['Train_label']
            # self.train_data = np.asarray(self.train_data)

        elif self.split == 'Valing':
            number_class = len(self.val_data_h5['Val_label'])
            self.val_data = self.val_data_h5['Val_pixel']
            self.val_labels = self.val_data_h5['Val_label']
            # self.val_data = np.asarray(self.val_data)

        else:
            number_class = len(self.test_data_h5['Test_label'])
            self.test_data = self.test_data_h5['Test_pixel']
            self.test_labels = self.test_data_h5['Test_label']
            # self.test_data = np.asarray(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            #import pdb
            #pdb.set_trace()
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Valing':
            img, target = self.val_data[index], self.val_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets and to return a PIL Image
        #train gray or rgb
        #img = Image.fromarray(img)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        import pdb
        #pdb.set_trace()
        #train gray of 3 channels
        # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img), axis=2)
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Valing':
            return len(self.val_data)
        else:
            return len(self.test_data)