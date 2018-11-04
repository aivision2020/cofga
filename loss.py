import torch
import torch.nn as nn
import numpy as np

class RankLoss(nn.Module):
    """
    input NxC dim values. prediction in 0,1 range, label is 0 or 1
    N - number of objects
    C - number of classes
    for every pair i,j in range(N) and class c
    RankLoss(i,j,c)= clip(prediction[j,c]+0.1-prediction[i,c], 0) if label[i,c]==1 and label[j,c]=0
    meaning that object of class c should have a higher score (with margin 0.1) than object not of that class
    """
    def __init__(self, margin=0.1):
        super(RankLoss, self).__init__()
        self.margin = margin
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, label):
        pred = self.sigmoid(pred)
        diff = pred[None,:,:]-pred[:,None,:]+self.margin
        gt_diff = torch.clamp(label[:,None,:]-label[None,:,:],0,1)
        loss = torch.clamp(diff*gt_diff,0,1)
        loss = loss.sum(0).sum(0)
        denom=gt_diff.sum(0).sum(0).clamp(min=1)
        return loss/denom

    def forward_(self, pred, label):
        loss = np.zeros_like(label[0],dtype=np.float32)
        for i in range(pred.shape[0]):
            for j in range(pred.shape[0]):
                for c in range(pred.shape[1]):
                    if label[i,c]==1 and label[j,c]==0:
                        loss[c]+=max(0, pred[j,c]+self.margin-pred[i,c])
        return loss

