# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm

import tensorflow # needs to call tensorflow before torch, otherwise crush
sys.path.append('Utils')
from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('Dataset')
from JigsawNetwork import Network

from TrainingUtils import adjust_learning_rate, compute_accuracy


parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet') # 모델에 사용되는 하이퍼파라미터 정의
parser.add_argument('data', type=str, help='Path to Imagenet folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=1000, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', #action='store_true'를 사용하여 해당하는 인자(argument)가 입력되면 True, 입력되지 않으면 False로 인식하게 됩니다.
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

#from ImageDataLoader import DataLoader
from JigsawImageLoader import DataLoader


def main():
    if args.gpu is not None: #GPU 없을 때.
        print(('Using GPU %d'%args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    else:
        print('CPU mode')
    
    print('Process number: %d'%(os.getpid()))
    
    trainpath = args.data+'/ILSVRC2012_img_train' ## DataLoader initialize ILSVRC2012_train_processed 초기화,,,
    if os.path.exists(trainpath+'_255x255'): #해당 경로에 파일 존재한다면 경로에 _255x255 문자열 더해주기.
        trainpath += '_255x255'
    train_data = DataLoader(trainpath,args.data+'/ilsvrc12_train.txt', #지정한 데이터 경로에 있는 데이터들을 train_data로. dataloader의 인자로 설정함. data_path, txt_list, classes=1000 -> 기본
                            classes=args.classes) #클래스의 수, 디폴트값은 1000
    train_loader = torch.utils.data.DataLoader(dataset=train_data, #데이터, 배치사이즈, 셔플 o, cpu 넘버
                                            batch_size=args.batch,
                                            shuffle=True,
                                            num_workers=args.cores) #데이터셋에 성공적으로 접근했으니, 이제 데이터셋을 torch.utils.data.DataLoader 로 넘겨줍니다. DataLoader 는 데이터셋을 sampler와 조합시켜 데이터셋을 순회할 수 있는 iterable을 만들어줍니다.
    
    valpath = args.data+'/ILSVRC2012_img_val' #검증셋 
    if os.path.exists(valpath+'_255x255'): #255x255 문자열 더해주기
        valpath += '_255x255'
    val_data = DataLoader(valpath, args.data+'/ilsvrc12_val.txt', #지정한 데이터 경로에 있는 데이터들을 train_data로. dataloader의 인자로 설정함.
                            classes=args.classes)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, #데이터, 배치사이즈, 셔플 o, cpu 넘버
                                            batch_size=args.batch,
                                            shuffle=True,
                                            num_workers=args.cores)
    N = train_data.N #train데이터 개수 -> 데이터 경로 imagenet으로 두고 출력했을 때 1281167
    
    iter_per_epoch = train_data.N/args.batch #에포크: 전체 train data / 배치사이즈, 10009 나옴. -> 배치사이즈 128로 설정했었음. iteration
    print('Images: train %d, validation %d'%(train_data.N,val_data.N)) # train 데이터 개수, val 데이터 개수 출력
    
    # Network initialize 네트워크 초기화
    net = Network(args.classes)
    if args.gpu is not None:
        net.cuda()
    
    ############## Load from checkpoint if exists, otherwise from model ###############
    if os.path.exists(args.checkpoint): #체크포인트에서 진행사항 저장
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        if len(files)>0:
            files.sort()
            #print files
            ckp = files[-1]
            net.load_state_dict(torch.load(args.checkpoint+'/'+ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            print('Starting from: ',ckp)
        else:
            if args.model is not None:
                net.load(args.model)
    else:
        if args.model is not None:
            net.load(args.model)

    criterion = nn.CrossEntropyLoss() #크로스 엔트로피 -> 분류 문제에 주로 쓰임.  정답과 예측값 사이의 유사정도를 파악 할 수 있는 loss
    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4) #최적화함수 SGD
    #SGD(Stochastic Gradient Descent) 는 전체데이터 중 랜덤하게 선택된 단 하나의 데이터(배치사이즈=1)를 이용하여 학습시키는 경사하강법
    
    logger = Logger(args.checkpoint+'/train')
    logger_test = Logger(args.checkpoint+'/test')
    
    ############## TESTING ###############
    if args.evaluate: #EVALUATE: VALID 세트에 대한 모델 평가, TRAIN 아님 (위에 하이퍼파라미터 참고) -> 값이 true 일 때 조건문 실행. (위에 체크포인트에서 평가 됐는지 여부 보는 듯)
        test(net,criterion,None,val_loader,0)
        return
    
    ############## TRAINING ###############
    print(('Start training: lr %f, batch size %d, classes %d'%(args.lr,args.batch,args.classes)))
    print(('Checkpoint: '+args.checkpoint))
    
    # Train the Model
    batch_time, net_time = [], []
    steps = args.iter_start #Starting iteration count
    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs): #에포크 기본:70. 
        if epoch%10==0 and epoch>0:
            test(net,criterion,logger_test,val_loader,steps)
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        
        end = time()
        for i, (images, labels, original) in enumerate(train_loader):
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]
            
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad() #Neural Network에서 parameter들을 업데이트 할때 zero_grad()를 사용, 루프가 한번 돌고나서 역전파를 하기전에 반드시 zero_grad()로 .grad 값들을 0으로 초기화
            t = time()
            outputs = net(images)
            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]
            
            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5)) # output, target, topk=(1,) (파라미터)
            acc = prec1[0]

            loss = criterion(outputs, labels)
            loss.backward() #loss.backwards()를 호출하여 예측 손실(prediction loss)을 역전파합니다. PyTorch는 각 매개변수에 대한 손실의 변화도를 저장합니다.
            optimizer.step() #변화도를 계산한 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정(가중치 갱신)
            loss = float(loss.cpu().data.numpy())

            if steps%20==0:
                print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' %(
                            epoch+1, args.epochs, steps, 
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss,acc)))

            if steps%20==0:
                logger.scalar_summary('accuracy', acc, steps)
                logger.scalar_summary('loss', loss, steps)
                
                original = [im[0] for im in original]
                imgs = np.zeros([9,75,75,3])
                for ti, img in enumerate(original):
                    img = img.numpy()
                    imgs[ti] = np.stack([(im-im.min())/(im.max()-im.min()) 
                                         for im in img],axis=2)
                
                logger.image_summary('input', imgs, steps)

            steps += 1

            if steps%1000==0:
                filename = '%s/jps_%03i_%06d.pth.tar'%(args.checkpoint,epoch,steps)
                net.save(filename)
                print('Saved: '+args.checkpoint)
            
            end = time()

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

def test(net,criterion,logger,val_loader,steps):
    print('Evaluating network.......')
    accuracy = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5)) #pytorch 에서의 input tensor 에서 주어진 k 값에 따라 가장 큰 값 k 개를 찾는 함수이다. image classification 에서 train/validation 시 accuracy 측정할 때 자주 사용된다.

        accuracy.append(prec1[0])

    if logger is not None:
        logger.scalar_summary('accuracy', np.mean(accuracy), steps)
    print('TESTING: %d), Accuracy %.2f%%' %(steps,np.mean(accuracy)))
    net.train()

if __name__ == "__main__":
    main()
