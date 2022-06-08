# -*- coding: utf-8 -*-
#이미지의 크기를 256으로 조정하고 가로 세로 비율을 유지하며 
#중앙에 255x255 크기의 패치를 자르는 produce_small_data.py를 사용하여 
#이미지를 사전 처리할 수 있습니다.
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

datapath = 'path-to-imagenet' #경로 수정 필요. ex)datapath = '/home/pc1/data/torrents/'

trainval = 'train'
#trainval = 'val'

def main():
    #data = DataLoader(datapath+'/ILSVRC2012_img_train', datapath+'/ilsvrc12_train.txt')
    data = DataLoader(datapath+'/ILSVRC2012_img_'+trainval, datapath+'/ilsvrc12_'+trainval+'.txt') #파일명 수정
    loader = torch.utils.data.DataLoader(dataset=data,batch_size=1, 
                                        shuffle=False,num_workers=20)
    
    count = 0
    for i, filename in enumerate(tqdm(loader)):
        count += 1


class DataLoader(data.Dataset):
    def __init__(self,data_path,txt_list):
        self.data_path = data_path if data_path[-1]!='/' else data_path[:-1]
        self.names, _ = self.__dataset_info(txt_list)
        self.__image_transformer = transforms.Compose([ #256으로 사이즈 조절, 
                            transforms.Resize(256,Image.BILINEAR),
                            transforms.CenterCrop(255)]) #중앙에 255x255 크기의 패치를 자르는 produce_small_data.py를 사용하여 이미지 사전처리
        self.save_path = self.data_path+'_255x255/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for name in self.names:
            if '/' in name:
                fold = self.save_path+name[:name.rfind('/')]
                if not os.path.exists(fold):
                    os.makedirs(fold)
    
    def __getitem__(self, index):
        name = self.names[index]
        if os.path.exists(self.save_path+name):
            return None, None
        
        filename = self.data_path+'/'+name
        img = Image.open(filename).convert('RGB')
        img = self.__image_transformer(img)
        img.save(self.save_path+name)
        return self.names[index]


    def __len__(self):
        return len(self.names)
    
    def __dataset_info(self,txt_labels):
        with open(txt_labels,'r') as f:
            images_list = f.readlines()
        
        file_names = []
        labels     = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))
            #if len(file_names)>128*10:
                #break
        
        return file_names, labels

if __name__ == "__main__":
    main()
