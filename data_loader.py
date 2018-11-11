import yaml
import torch.nn as nn
import PIL
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
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import ipdb
from sklearn.metrics import average_precision_score as MAP
import copy

config_file = 'data/config.yaml'

def create_config_file(csv_file_name):
    dat = pd.read_csv(csv_file_name)
    #dat = dat.rename(str.strip, axis='columns')
    #dat = dat.rename(lambda x : x.replace(' ',''), axis='columns')

    remap_keys = ['general_class', 'sub_class','color']
    remap={}
    for k in dat.keys()[-15:]:
        vals=np.unique(dat[k])
        d = {}
        for i,v in enumerate(vals):
            #dat[dat==v]=i
            if type(v) is np.int64:
                d[str(v)]=int(max(0,v))
            else:
                assert type(v) is str, (k,v,type(v))
                d[v]=i
        remap[k]=d
    yaml.dump(remap, open('data/config.yaml','w' ))

class PredictionCollector(object):
    def __init__(self, keys, criterion=None):
        self.output = pd.DataFrame(columns=keys)
        self.labels = pd.DataFrame(columns=keys)
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.BCELoss(reduction='none')

    def add(self, tagids, predictions,labels=None):
        vals, inds = np.unique(tagids, return_index=True)
        tmp = pd.DataFrame(index=vals,columns=self.output.keys())
        tmp.loc[vals]=predictions[inds]
        self.output = self.output.append(tmp)
        if labels is not None:
            tmp = pd.DataFrame(index=vals,columns=self.output.keys())
            tmp.loc[vals]=labels[inds]
            self.labels = self.labels.append(tmp)

    # MAFAT version of precision
    def precision_np_metric_label(self, preds, targs, epsilon=1e-8):
        preds_indexes =  np.argsort(preds)[::-1]
        targs_sorted = targs[preds_indexes]
        preds_sorted = preds[preds_indexes]
        p_at_k = np.zeros_like(preds_sorted)
        tp = 0.0
        for i in range(len(preds_sorted)):
            if targs_sorted[i]>0.5:
                tp+=1
                p_at_k[i] = tp / (i + 1)
        ret = p_at_k.sum() / (tp+epsilon)
        assert not(ret==0 and np.sum(targs)>0)
        return ret

    def precision_np_metric(self, preds, targs, epsilon=1e-8):
        p_at_ks = np.zeros(targs.shape[-1])
        for i in range(targs.shape[-1]):
            p_at_ks[i] = self.precision_np_metric_label(preds[:, i], targs[:, i])
        return p_at_ks

    def calc_map(self):
        y_true = self.labels.values.astype(int)
        y_score = self.output.values.astype(float)
        return self.precision_np_metric(y_score, y_true), self.labels.keys(), self.labels.values.sum(axis=0)

    def calc_loss(self):
        ret = self.criterion(torch.from_numpy(self.output.values.astype(float)), torch.from_numpy(self.labels.values.astype(float))).mean(0)
        assert len(ret)==37
        return ret

    def save(self, csv_filename, sanitize=False):
        output = copy.copy(self.output)
        if sanitize:
            p_large = output['large vehicle']/(output['large vehicle']+output['small vehicle'])
            output['large vehicle']=p_large
            output['small vehicle']=(1-p_large)
            sub_large = 'Truck, Light truck, Cement mixer, Dedicated agricultural vehicle, Crane truck, Prime mover, Tanker, Bus, Minibus'.split(',').strip().lower()
            sub_small = 'Sedan, Hatchback, Minivan, Van, Pickup truck, Jeep'.split(',').strip().lower()
            for k in sub_large:
                output[k] = output[k]*output['large vehicle']
            for k in sub_small:
                output[k] = output[k]*output['small vehicle']

            sub_small_features = 'Sunroof, Luggage carrier, Spare wheel'.split(',').strip().lower()
            sub_large_features = 'AC vents, Enclosed box, Ladder, Flatbed, Soft shell box, Harnessed to a cart'.split(',').strip().lower()

            for k in sub_large_features:
                output[k] = output[k]*output['large vehicle']
            for k in sub_small_features:
                output[k] = output[k]*output['small vehicle']

        by_prob = pd.DataFrame(index=self.output.index,columns=output.keys())
        for key in by_prob.keys():
            inds = np.argsort(output[key])[::-1]
            assert np.all(inds>=0)
            by_prob[key] = output.index[inds].copy()


        by_prob.to_csv(csv_filename, index=False)
        return by_prob

class MafatDataset(Dataset):
    def __init__(self, csv_file_name, answer_csv, imfolder, preload=False,
            normalize_size=True, normalize_rotation=False, patch_size=128,
            boarder_ratio=2, mask_detection=False, augment=True,
            start=0, end=1, imageids=None):
        """
        croping scheme:
        patch_size is the size of pixels containing data.
        resize False will simply crop a patch of patch_size around the center of the detections
        resize True will take the detection+boarder_size pixels and resize to patch_size
        mask_detection will block the detection and leave only the boarder (for context)
        """
        super(MafatDataset, self).__init__()
        self.patch_size=patch_size
        self.boarder_ratio=boarder_ratio
        self.mask_detection=mask_detection
        self.normalize_rotation = normalize_rotation
        self.normalize_size = normalize_size
        rotation_jitter = 1 if normalize_rotation else 180
        if augment:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomRotation(rotation_jitter, PIL.Image.BILINEAR),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                ])
        if self.normalize_size:
            self.trans_final = transforms.Compose([
                transforms.Resize(self.patch_size, PIL.Image.BILINEAR),
                transforms.ToTensor()
                ])
        else:
            self.trans_final = transforms.Compose([
                transforms.ToTensor()])

        self.standardize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        self.top_class = ['general_class', 'sub_class', 'color']
        self.dat = pd.read_csv(csv_file_name)
        self.answer = pd.read_csv(answer_csv)

        self.imfolder = os.path.join(imfolder, '%d.*')
        assert os.path.exists('data/config.yaml')
        self.remap = yaml.load(open('data/config.yaml','r' ))

        self.ims={}

        if imageids is None:
            imageids = np.unique(self.dat['image_id'])
        if start is not None and end is not None:
            imageids = imageids[int(len(imageids)*start):int(len(imageids)*end)]
        self.dat = self.dat.loc[np.isin(self.dat['image_id'], imageids)]
        if preload:
            for id in imageids:
                imfile = self.imfolder%id
                imfile = glob.glob(imfile)[0]
                assert id not in self.ims
                self.ims[id] = cv2.cvtColor(cv2.imread(imfile), cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.dat)

    def encode(self, n, i):
        vec = np.zeros(n)
        vec[i]=1
        return vec

    def row_to_label(self, row):
        labels = pd.DataFrame(columns=self.answer.keys())
        labels.loc[0] = np.zeros(len(labels.keys()))

        for k,d in self.remap.items():
            if k in row:
                value = row[k]
                if k in self.top_class:
                    labels[value].loc[0]=1
                else:
                    labels[k].iloc[0]=int(value>0)

        return labels.values.ravel().astype(np.float32)

    def get_class_names(self):
        return self.answer.keys()

    def labels_to_text(self, labels):
        return ', '.join([self.answer.keys()[i] for i,v in enumerate(labels) if v>0.5])

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
        index = int(index) #sometimes, when using sampler, this comes in a Tensor. Why?
        row = self.dat.iloc[index]
        imageid = row['image_id']#.values[0]
        if imageid in self.ims:
            im = self.ims[imageid]
        else:
            imfile = self.imfolder%imageid
            print 'warning, loading image from HDD. Slow!', imfile
            imfile = glob.glob(imfile)[0]
            im = cv2.cvtColor(cv2.imread(imfile), cv2.COLOR_BGR2RGB)
        xs = ['p1_x', ' p2_x', ' p3_x', ' p4_x', 'p1_x']
        ys = ['p_1y', ' p2_y', ' p3_y', ' p4_y', 'p_1y']

        points = np.vstack((row[xs],row[ys])).astype(float)
        diff = np.diff(points, axis=1)
        lens = np.linalg.norm(diff, axis=0)
        i = np.argmax(lens)
        dx,dy = diff[:, i]
        detection_size = int(np.sqrt(dx**2+dy**2))
        patch_size = detection_size*self.boarder_ratio
        if not self.normalize_size:
            patch_size = np.maximum(self.patch_size, patch_size)

        tmp_half_size = int(1.5*patch_size/2)
        center_x = np.mean(row[xs])
        center_y = np.mean(row[ys])
        I = self.crop(im,int(center_x),int(center_y),tmp_half_size)
        #print I.shape
        I = self.transforms(I)

        if self.normalize_rotation:
            I = F.rotate(I, np.rad2deg(np.arctan2(dy,dx)), PIL.Image.BILINEAR)
        y = int(tmp_half_size+np.random.rand()*5)
        x = int(tmp_half_size+np.random.rand()*5)
        half_size = int(patch_size/2)
        I = F.crop(I, y-half_size, x-half_size, patch_size, patch_size)

        if self.mask_detection:
            half_detection=int(detection_size/2)
            y = int(I.shape[1]/2)
            x = int(I.shape[2]/2)
            img = np.asarray(I)
            img[:,y-half_detection:y+half_detection, x-half_detection:x+half_detection]=0
            I=Image.fromarray(img)

        I = self.trans_final(I)
        I = self.standardize(I)
        labels = self.row_to_label(row)
        # assert I.shape[-1]==224, (I.shape)
        # assert I.shape[-2]==224, (I.shape, index)
        assert I.shape[-1] > 0 and I.shape[-2] > 0, (I.shape, index)

        return (I, labels, '%d,%d,%s'%(row['tag_id'],imageid,self.labels_to_text(labels)))

    def get_class(self, class_name, class_value, num=-1):
        mask = self.dat[class_name]==class_value
        num = np.minimum(num, np.count_nonzero(mask))
        trans = transforms.ToPILImage()
        samples = np.where(mask)[0]
        return [np.array(trans(self.__getitem__(i)[0])) for i in samples[:num]], self.dat['image_id'][mask]

    def get_weights(self):
        if os.path.exists('data/train_features.npy'):
            print 'reading from file'
            features = np.load('data/train_features.npy')
        else:
            features = [self.row_to_label(self.dat.iloc[i]) for i in range(len(self.dat))]
            features = np.array(features).T
            np.save('data/train_features.npy', features)
        occurences = features.sum(axis=1,keepdims=True)
        weights =np.minimum(np.sum(features/occurences,axis=0),0.01)#no more than 1%
        assert len(occurences) == 37
        assert len(weights) == len(self.dat)
        return weights, len(self.dat)


def create_train_val_dataset(csv_file_name, answer_csv, imfolder, split=0.8, image_group_file=None, augment=True,**kwargs):
    if image_group_file is None:
        train = MafatDataset(csv_file_name, answer_csv, 'data/training imagery', start=0, end=0.8, augment=augment, **kwargs)
        val =  MafatDataset(csv_file_name, answer_csv , 'data/training imagery', start=0.8, end=1, augment=False, **kwargs)
        return train, val
    im_groups = yaml.load(open(image_group_file))
    all_ids=np.hstack(im_groups)
    N = len(all_ids)
    train_ims = np.hstack([s for s in im_groups if len(s)>1])
    the_rest = np.hstack([s for s in im_groups if len(s)==1])
    #np.random.shuffle(the_rest)
    train_ims = np.concatenate((train_ims, the_rest[:int(N*split-len(train_ims))]))
    val_ims = [im for im in set(all_ids).difference(train_ims)]
    assert len(set(val_ims).intersection(train_ims))==0
    assert len(val_ims)+len(train_ims)==len(all_ids)
    train = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', imageids=train_ims, augment=augment, **kwargs)
    val = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', imageids=val_ims, augment=False, **kwargs)
    return train, val


