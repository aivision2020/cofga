import pandas as pd
import torchvision.transforms as transforms
from data_loader import MafatDataset
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--category', default = 'color', help='what class/category')
parser.add_argument('--answer',  help='answer csv file (ranking)')

parser.add_argument('-n', default = 1, type=int, help='how many examples to show. -1 to show all')
args = parser.parse_args()
dataset = MafatDataset('data/test.csv', 'data/answer.csv', 'data/test imagery',
        False, resize=False, patch_size=224, augment=False)
dat = pd.read_csv(args.answer)

ids = dat[args.category][:args.n]
trans = transforms.ToPILImage()

ims = [np.array(trans(dataset.__getitem__(np.where(dataset.dat['tag_id']==i)[0][0])[0]))
            for i in ids]
n = int(np.sqrt(len(ims)))
rows = [np.hstack(ims[i*n:(i+1)*n]) for i in range(n)]
grid = np.vstack(rows)
plt.imshow(grid)
plt.show()
