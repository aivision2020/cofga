from data_loader import MafatDataset
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--category', default = 'color', help='what class/category')
parser.add_argument('--value', default = 1, help='what value')
parser.add_argument('--test', action='store_true')
parser.add_argument('-n', default = 1, type=int, help='how many examples to show. -1 to show all')
args = parser.parse_args()
dataset = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery',
        False, resize=False, patch_size=224, augment=False)

if args.category not in dataset.get_class_names():
    print 'unknown category %s. try '%args.category, dataset.get_class_names()
    exit()
print 'unknown category %s. try '%args.category, dataset.get_class_names()
ims, im_ids = dataset.get_class(args.category, args.value, args.n)
print im_ids
n = int(np.sqrt(len(ims)))
rows = [np.hstack(ims[i*n:(i+1)*n]) for i in range(n)]
print rows[0].shape
grid = np.vstack(rows)
plt.imshow(grid)
plt.show()
