import torch
import pandas as pd
import numpy as np
from data_loader import MafatDataset, PredictionCollector, create_config_file, create_train_val_dataset
import matplotlib.pyplot as plt

DISPLAY=True

def test_collector():
    dataset = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', preload=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)#args.workers)
    collector = PredictionCollector(dataset.get_class_names())
    for data in loader:
        images,labels, gt_text = data
        labels = labels.detach().numpy()
        ids = [int(text.split(',')[0]) for text in gt_text]
        collector.add(ids, labels)
        assert collector.output['dedicated agricultural vehicle'][24690]==1
        break
    assert len(collector.output) == 8
    by_prob = collector.save('data/answer_v0.csv')
    assert np.all(by_prob['small vehicle']!=by_prob['large vehicle'])
    assert np.all([len(np.unique(by_prob[key])) == by_prob.shape[0] for key in by_prob.keys()])
    print by_prob['dedicated agricultural vehicle']
    assert by_prob['dedicated agricultural vehicle'].iloc[0]==24690
    print by_prob

def test_loader():
    create_config_file('data/train.csv')
    #data = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', False, True)
    data, _ = create_train_val_dataset('data/train.csv', 'data/answer.csv', 'data/training imagery', False, True)
    print len(data)
    for i in [3977, 49, 300, 500]:
        im, l, text = data.__getitem__(i)
        print 'label', l
        print text
        assert len(l)==37 or len(l)==0
        assert im.shape[-1]==224
        if DISPLAY:
            plt.imshow(im[0],cmap='gray')
            plt.title('%d'%i)
            plt.show()

def test_loader_split():
    full = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', True)
    train = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', True, start=0, end=0.8)
    val = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', True, start=0.8, end=1)
    assert len(np.unique(full.dat['image_id'])) > len(np.unique(train.dat['image_id']))
    assert len(np.unique(full.dat['image_id'])) > len(np.unique(val.dat['image_id']))
    assert 0.78 < float(len(train.dat))/len(full.dat) < 0.82
    assert 0.18 < float(len(val.dat))/len(full.dat) < 0.22

if __name__=='__main__':
    test_loader()
    #test_collector()
    #test_loader_split()
