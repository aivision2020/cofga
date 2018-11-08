from disjoint_set import *
import os
import numpy as np
import pandas as pd
from data_loader import MafatDataset, PredictionCollector, create_train_val_dataset

features_keys = 'general_class   sub_class   sunroof luggage_carrier open_cargo_area enclosed_cab    spare_wheel wrecked flatbed ladder  enclosed_box soft_shell_box  harnessed_to_a_cart ac_vents    color'.split()
xs = ['p1_x', ' p2_x', ' p3_x', ' p4_x']
ys = ['p_1y', ' p2_y', ' p3_y', ' p4_y']

from collections import namedtuple
Match = namedtuple("Match", "i1, i2, dx, dy")
def compare_ims(detections1, detections2, encoding1, encoding2):
    assert len(detections1)==len(encoding1)
    assert len(detections2)==len(encoding2)
    matches=[]
    for i1, (_, det1) in enumerate(detections1.iterrows()):
        det1x = det1[xs]
        det1y = det1[ys]
        det1_features=det1[features_keys]
        for i2, (_, det2) in enumerate(detections2.iterrows()):
            if encoding1[i1]!=encoding2[i2]:
                continue
            dx = det1x-det2[xs]
            dy = det1y-det2[ys]
            if np.all(det1_features==det2[features_keys]) and dx.max()-dx.min()<5 and dy.max()-dy.min()<5:
                matches.append(Match(i1, i2, dx.mean(),dy.mean()))

    best_inliers = 0
    for i, m in enumerate(matches):
        inliers = np.count_nonzero([np.abs(m2.dx-m.dx) < 10 and np.abs(m2.dy-m.dy)<5 for m2 in matches])
        if inliers>best_inliers:
            best_inliers=inliers
            best_ind = i

    if best_inliers >= 2:
        print 'found matching images '
        return matches[best_ind].dx, matches[best_ind].dy

    return None

dataset = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', preload=False)
train = dataset.dat
print np.unique(train[features_keys])

if os.path.exists('/tmp/features.npy'):
    print 'reading from file'
    features_ = np.load('/tmp/features.npy')
else:
    features = [dataset.row_to_label(train.iloc[i]) for i in range(len(train))]
    features_ = np.array(features).T
    np.save('/tmp/features.npy', features_)

#every detection is now reduced to a singe index "encoding". val is the deteciton, but we don't care about that
val, encoding = np.unique(features_, axis=1, return_inverse=True)
for i,r in enumerate(encoding): assert np.all(features_[:,i]==val[:,r])
print 'there are %d unique detections'%encoding.max()

#brute force match all pairs ~ 1 million comparisons. I hope this doesn't take too long
im_ids = np.unique(train['image_id'])
dset = DisjointSet(im_ids.tolist())
for im1 in range(len(im_ids)):
    print im_ids[im1]
    if im_ids[im1]!=12193:
        continue

    detect_im1 = np.where(train['image_id']==im_ids[im1])[0]
    detections1 = train.iloc[detect_im1]
    for im2 in range(im1+1, len(im_ids)):
        if im_ids[im2]!=15850:
            continue
        import ipdb; ipdb.set_trace()
        if dset.find(im_ids[im2])==dset.find(im_ids[im1]):
            continue
        detect_im2 = np.where(train['image_id']==im_ids[im2])[0]
        detections2 = train.iloc[detect_im2]
        shift = compare_ims(detections1, detections2, encoding[detect_im1], encoding[detect_im2])
        if shift is not None:
            print 'found pair ', im_ids[im1], im_ids[im2]
            dset.union(im_ids[im1], im_ids[im2])

yaml.dump(dset._disjoint_set, open('data/train_pairs.yaml','w'))
import ipdb; ipdb.set_trace()
