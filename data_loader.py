import shutil
import itertools
import numpy as np
import torch
import torch.utils.data
import cv2
import glob
import os
import pandas as pd
import yaml
from torch.autograd import Variable
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
from torch.utils.data import Dataset

config_file = 'data/config.yaml'
class MafatDataset(Dataset):
    def __init__(self, csv_file_name, preload=False):
        super(MafatDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ])
        self.dat = pd.read_csv(csv_file_name)
        self.dat = self.dat.rename(str.strip, axis='columns')
        self.dat = self.dat.rename(lambda x : x.replace('_',''), axis='columns')
        self.dat = self.dat.rename(lambda x : x.replace(' ',''), axis='columns')

        self.imfolder = os.path.join(os.path.dirname(csv_file_name), 'imagery/%d.*')

        remap_keys = ['generalclass', 'subclass','color']
        self.remap={}
        for k in self.dat.keys()[-15:]:
            vals=np.unique(self.dat[k])
            d = {}
            for i,v in enumerate(vals):
                #self.dat[self.dat==v]=i
                if type(v) is np.int64:
                    d[str(v)]=int(max(0,v))
                else:
                    assert type(v) is str
                    d[v]=i
            self.remap[k]=d
        yaml.dump(self.remap, open('data/config.yaml','w' ))
        self.remap = yaml.load(open('data/config.yaml','r' ))
        #print self.remap
        #print self.dat
        for k,sub_dict in self.remap.items():
            for val, sub_class in sub_dict.items():
                col = self.dat[k]
                col[col==sub_class]=val
        self.dat.to_csv('data/train_clean.csv')
        self.ims={}

        if preload:
            imageids = np.unique(self.dat['imageid'])
            for id in imageids:
                imfile = self.imfolder%id
                imfile = glob.glob(imfile)[0]
                assert id not in self.ims
                self.ims[id]=cv2.imread(imfile)
                self.ims[id] = cv2.cvtColor(cv2.imread(imfile), cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.dat)

    def encode(self, n, i):
        vec = np.zeros(n)
        vec[i]=1
        return vec

    def row_to_label(self, row):
        top_class = '''generalclass subclass color'''.split()
        '''sunroof luggage_carrier open_cargo_area
        enclosed_cab    spare_wheel wrecked flatbed ladder  enclosed_box
        soft_shell_box  harnessed_to_a_cart ac_vents    color'''
        #for  row.values.ravel()[-15:]
        label = []
        for k,d in self.remap.items():
            value = row[k].values[0]
            if k in top_class:
                label.append(self.encode(len(np.unique(d.values())), d[value]))
            else:
                label.append([int(int(value)>0)])

        labels = np.hstack(label).astype(np.float32)
        return labels

    def get_class_names(self):
        top_class = '''generalclass subclass color'''.split()
        text = []
        for k,d in self.remap.items():
            if k in top_class:
                text.extend(d.keys())
            else:
                text.append(k)
        return text

    def labels_to_text(self, labels):
        top_class = '''generalclass subclass color'''.split()
        counter = 0
        text = []
        for k,d in self.remap.items():
            if k in top_class:
                n = len(np.unique(d.values()))
                value = np.argmax(labels[counter:counter+n])
                counter+=n
                name = [name for name,v in d.items() if v == value]
                text.extend(name)
            else:
                value = labels[counter]
                counter+=1
                if value > 0.5:
                    text.append(k)
        assert counter==len(labels)
        return ', '.join(text)

    def crop(self, im, x, y, size=112):
        if size<x<im.shape[1]-size and size<y<im.shape[0]-size:
            return im[y-size:y+size, x-size:x+size, :]
        else:
            patch = np.zeros((size*2,size*2,3),dtype=np.uint8)
            size_x1 = min(size, x)
            size_x2 = min(size, im.shape[1]-x)
            size_y1 = min(size, y)
            size_y2 = min(size, im.shape[0]-y)
            patch[size-size_y1:size+size_y2,size-size_x1:size+size_x2,:] = im[y-size_y1:y+size_y2,x-size_x1:x+size_x2,:]
            return patch

    def __getitem__(self, index):
        row = self.dat.iloc[[index]]
        print row
        imageid = row['imageid'].values[0]
        if imageid in self.ims:
            im = self.ims[imageid]
        else:
            imfile = self.imfolder%imageid
            imfile = glob.glob(imfile)[0]
            im = cv2.cvtColor(cv2.imread(imfile), cv2.COLOR_BGR2RGB)
            
        xs = ['p%dx'%i for i in range(1,5)]
        ys = ['p%dy'%i for i in range(1,5)]

        x = np.mean(row[xs].values)
        y = np.mean(row[ys].values)
        I = self.crop(im,int(x),int(y))
        I = self.transforms(I)

        labels = self.row_to_label(row)

        return (I, labels, '%d,%d,%s'%(index,imageid,self.labels_to_text(labels)))


if __name__=='__main__':
    data = MafatDataset('data/train/train.csv')
    print len(data)
    for i in [49]:
        im, l, text = data.__getitem__(i)
        print 'label', l
        print text
        assert len(l)==37
        import ipdb; ipdb.set_trace()
        import matplotlib.pyplot as plt
        plt.imshow(im[0])
        plt.show()
