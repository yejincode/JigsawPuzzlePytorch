# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms 
# transform 을 해서 데이터를 조작하고 학습에 적합하게 만들어야 합니다.
# PyTorch의 torchvision 라이브러리는 transforms 에서 다양한 변환 기능을 제공합니다. 
# transform을 사용하여 데이터의 일부 조작을 수행하고 훈련에 적합하게 만들 수 있습니다.
from PIL import Image #이미지 분석 및 처리 쉽게 하도록 하는 라이브러리(마스킹, 투명도, 밝기 보정,...)


class DataLoader(data.Dataset):
    def __init__(self, data_path, txt_list, classes=1000):
        self.data_path = data_path
        self.names, _ = self.__dataset_info(txt_list)
        self.N = len(self.names)
        self.permutations = self.__retrive_permutations(classes)

        self.__image_transformer = transforms.Compose([ #여러 단계로 변환해야 하는 경우, Compose를 통해 여러 단계를 묶을 수 있다.
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(), #데이터를 tensor로 바꿔준다. 텐서란 매우 수학적인 개념으로 데이터의 배열이라고 볼 수 있습니다. 텐서의 Rank는 간단히 말해서 몇 차원 배열인가를 의미합니다
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index] #데이터경로

        img = Image.open(framename).convert('RGB') #이미지 변환
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut shortcut 피하기 위해 패치 정규화시킴. 
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1 #표준편차 1? 
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels): #데이터셋 정보. 파일이름과 라벨을 나눔.
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrive_permutations(self, classes): #순열
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def rgb_jittering(im): #color jitter은 이미지 data augmentation 기법의 하나로, 이미지의 색상, 채도, 명도 등을 임의로 변환함. 
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8') #데이터타입변경
