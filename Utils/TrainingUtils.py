# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:53:30 2017

@author: bbrattol
"""

# lr 조정
def adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=30, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay ** (epoch // step))
    print('Learning Rate %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# 정확도 측정 #top 5 error... 1000개의 클래스 다 안보기 /그 중 k개의 정확도 보겠다. 
def compute_accuracy(output, target, topk=(1,)): #k는 5
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk) #topk 출력: (value,indices) 튜플/ indices는 출력된 value의 인덱스
    batch_size = target.size(0) 

    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t() #matrix를 대각선으로 바꿔줌. (table의 행과 열을 바꿔줌)
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #비교 연산 메서드. eq()는 == / view는 텐서 reshape시켜줌./expand_as 입력한 텐서 확장시킴. 

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

