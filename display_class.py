from data_loader import MafatDataset, create_train_val_dataset
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--category', default = 'color', help='what class/category')
parser.add_argument('--value', default = 1, help='what value')
parser.add_argument('--test', action='store_true')
parser.add_argument('-n', default = 1, type=int, help='how many examples to show. -1 to show all')
args = parser.parse_args()
train, val = create_train_val_dataset('data/train.csv', 'data/answer.csv',
        'data/training imagegy', image_group_file='data/train_image_groups.yaml',
        preload=False, augment=False)

if args.category not in train.get_class_names():
    print 'unknown category %s. try '%args.category, train.get_class_names()
    exit()
for name, dataset in zip(['val','train'],[val, train]):
    ims, im_ids = dataset.get_class(args.category, args.value, args.n)
    print 'found %d %s images'%(len(ims), args.category)
    print im_ids
    n = int(np.sqrt(len(ims)))
    rows = [np.hstack(ims[i*n:(i+1)*n]) for i in range(n)]
    print rows[0].shape
    grid = np.vstack(rows)
    plt.figure()
    plt.imshow(grid)
    #for im, id in zip(ims, im_ids):
    #    plt.figure()
    #    plt.imshow(im)
    #    plt.title('%s_%d'%(name,id))
plt.show()
    
