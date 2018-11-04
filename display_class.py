from data_loader import MafatDataset
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--category', default = 'color', help='what class/category')
parser.add_argument('--value', default = 1, help='what value')
parset.add_argument('--test', action='store_true')
parser.add_argument('-n', default = 1, type=int, help='how many examples to show. -1 to show all')
args = parser.parse_args()
dataset = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery',
        False, resize=False, patch_size=224, augment=False)

ims = dataset.get_class(args.category, args.value, args.n)
n = int(np.sqrt(len(ims)))
rows = [np.hstack(ims[i*n:(i+1)*n]) for i in range(n)]
print rows[0].shape
grid = np.vstack(rows)
plt.imshow(grid)
plt.show()
