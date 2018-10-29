import torch
import numpy as np
from loss import RankLoss
from torch.autograd import Variable

def close(a,b):
    return np.abs(a-b)<0.001

def test_rank_loss():
    GT = np.array([[0,1]]).T #2 samples. one class
    pred = np.array([[0.4,0.5]]).T
    criteria = RankLoss()
    loss = criteria(pred,GT)
    assert len(loss)==1
    assert close(loss,0), loss

    pred = np.array([[0.45,0.5]]).T
    loss = criteria(pred,GT)
    assert close(loss,0.05), loss


    GT = np.array([[0,1],[1,0]]).T #2 samples. two class
    pred = np.array([[0.45,0.5],[0.5,0.4]]).T
    loss = criteria(pred,GT)
    assert len(loss)==2
    assert close(loss[0],0.05), loss[0]
    assert close(loss[1],0)


def test_rank_loss_torch():
    GT = torch.from_numpy(np.array([[0,1],[1,0]], dtype=np.float32).T) #2 samples. two class
    pred = torch.from_numpy(np.array([[0.45,0.5],[0.5,0.4]], dtype=np.float32).T)
    criteria = RankLoss()
    loss = criteria(pred,GT)
    assert len(loss)==2
    assert close(loss[0],0.05), loss[0]
    assert close(loss[1],0)
if __name__=='__main__':
    test_rank_loss_torch()

