import pandas as pd
import numpy as np
import os
import cv2
import argparse
import torchvision.models as models
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_loader import MafatDataset, PredictionCollector, create_train_val_dataset
from tensorboardX import SummaryWriter
from PIL import Image
from sklearn.metrics import average_precision_score as MAP
from loss import RankLoss

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', action='store_false')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--freeze', dest='freeze', action='store_true', help='freeze params for base model')
parser.add_argument('--weighted-loss', dest='weighted_loss', action='store_true', help='use weights in log loss')
parser.add_argument('--preload', dest='preload', action='store_false', help='preload all images into memory before training')
parser.add_argument('--tag', type=str, default='baseline', help='tag - name of this experiment')
parser.add_argument('--start-tag', type=str, default=None, help='start-tag - name of experiment use as pretrain')
parser.add_argument('--architect', type=str, default='resnet18', help='architecture to use')

args = parser.parse_args()

def display_images(X, text, pred_text, nrow=4):
    images = []
    for im, t1, t2 in zip(X,text, pred_text):
        I = (np.transpose(im.cpu().detach().numpy(), [1,2,0]).copy()*225).astype(np.uint8)
        I = cv2.resize(I,(500,500))
        y_loc = 35
        t1 = t1.split(',')
        t2 = t2.split(',')
        I = cv2.putText(I, t1[0].strip(), (10,y_loc), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
        y_loc+=35
        I = cv2.putText(I, t1[1].strip(), (10,y_loc), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
        y_loc+=35
        t1 = t1[2:]

        for char in set(t1).intersection(t2):
            I = cv2.putText(I, char.strip(), (10,y_loc), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))
            y_loc+=35
        for char in set(t1).difference(t2):
            I = cv2.putText(I, char.strip(), (10,y_loc), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0))
            y_loc+=35
        for char in set(t2).difference(t1):
            I = cv2.putText(I, char.strip(), (10,y_loc), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255))
            y_loc+=35

        #for t in [t1,t2]:
        #    for char in t.split(','):
        #        I = cv2.putText(I, char.strip(), (10,y_loc), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
        #        y_loc+=35

        images.append(torch.from_numpy(np.transpose(I,[2,0,1])))
    return make_grid(images, nrow=4)


def evaluate(model, dataset, output_file, writer, freq=10):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model.cuda()
    model.eval()
    sigmoid = nn.Sigmoid()
    collector = PredictionCollector(dataset.get_class_names())
    with torch.no_grad():
        for it, data in enumerate(train_loader):
            images,labels, gt_text = data
            X = Variable(images, volatile=True).cuda()
            outputs = model(X)
            prediction = sigmoid(outputs)
            ids = [int(text.split(',')[0]) for text in gt_text]
            collector.add(ids, prediction)

            print 'display iter ', it
            if it%freq==0:
                pred_text = map(dataset.labels_to_text, prediction.detach().cpu().numpy()[:16])
                grid=display_images(X[:16], gt_text[:16], pred_text[:16], nrow=4)
                writer.add_image('Test/images', grid, it)
        probs = collector.save(output_file)

def write_to_board(writer, collector, it, dataset, curr_batch,  stage='train'):
    """
    it = current iteration to log
    """
    prediction, images, Y, gt_text = curr_batch
    pred_text = map(dataset.labels_to_text, prediction.detach().cpu().numpy()[:16])

    average_prec, names = collector.calc_map()
    per_class =  dict(zip(np.array(names), average_prec))

    loss = collector.calc_loss()
    writer.add_scalars('Loss', {'total_%s'%stage: loss.mean()}, it)
    writer.add_scalars('Loss', {'%s_%s'%(name,stage):l
                for name,l in zip(names, loss)}, it)
    writer.add_scalars('MAP', {'total_%s'%stage: average_prec.mean()}, it)
    map_per_class = {'%s_%s'%(name,stage):per_class[name]
                for name,l in zip(names, loss) if name in per_class}
    writer.add_scalars('MAP', map_per_class , it)
    grid=display_images(images[:16], gt_text[:16], pred_text[:16], nrow=4)
    writer.add_image('%s/images'%stage, grid, it)

    for key, value in sorted(map_per_class.iteritems(), key=lambda (k,v): (v,k)):
        print "MAP %s: %0.2f" % (key, value)


def main():
    if args.architect=='resnet18':
        model = models.resnet18(pretrained=True)
    if args.architect=='resnet50':
        model = models.resnet50(pretrained=True)
    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False
    assert not args.weighted_loss, 'not yet supported'
    model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features, out_features=1024), nn.ReLU(),
            nn.Linear(in_features=1024, out_features=37))
    cofga_v0 = model.cuda()

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    sigmoid = nn.Sigmoid()
    #criterion = RankLoss()
    optimizer = optim.SGD(cofga_v0.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(cofga_v0.parameters(), lr=args.lr)

    filename = '%s_BCEloss37%s.checkpoint.pth.tar'%(args.architect, args.tag)
    if args.start_tag is not None:
        start_filename = '%s_BCEloss37%s.checkpoint.pth.tar'%(args.architect, args.start_tag)
    else:
        start_filename = filename

    start_epoch = 0
    if os.path.exists(start_filename) and args.resume:
        checkpoint = torch.load(start_filename)
        cofga_v0.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    if args.start_tag is not None:
        start_epoch=0


    writer = SummaryWriter('runs/%s/%s'%(args.architect,args.tag))
    if args.evaluate:
        output_file = 'answer_%s.csv'%args.tag
        data = MafatDataset('data/test.csv', 'data/answer.csv', 'data/test imagery', preload=True,augment=False)
        evaluate(model, data, output_file, writer)
        exit()

    train_dataset, val_dataset = create_train_val_dataset('data/test.csv', 'data/answer.csv', 'data/test imagery', preload=args.preload)
    #train_dataset = val_dataset #TODO AVRAM
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    for epoch in range(start_epoch, args.epochs):  # loop over the dataset multiple times
        collector = PredictionCollector(train_dataset.get_class_names())
        for i, data in enumerate(train_loader):
            images,labels, gt_text = data
            X = Variable(images).cuda()
            outputs = cofga_v0(X)
            Y = Variable(labels).cuda()
            ids = [int(text.split(',')[0]) for text in gt_text]
            predictions = sigmoid(outputs)
            collector.add(ids, predictions.detach().cpu().numpy(), Y.detach().cpu().numpy())

            loss = criterion(outputs, Y)
            s=loss.mean()
            optimizer.zero_grad()
            s.backward()
            optimizer.step()
            print 'epoch %d, i %d, loss %0.2f'%(epoch,i, s)

        curr_batch = (predictions, images, Y, gt_text)
        write_to_board(writer, collector, epoch, train_dataset, curr_batch, stage='train')

        with torch.no_grad():
            collector = PredictionCollector(val_dataset.get_class_names())
            cofga_v0.eval()
            for _, data in enumerate(val_loader):
                images,labels, gt_text = data
                X = Variable(images).cuda()
                outputs = cofga_v0(X)
                Y = Variable(labels).cuda()
                ids = [int(text.split(',')[0]) for text in gt_text]
                predictions = sigmoid(outputs)
                collector.add(ids, predictions.detach().cpu().numpy(), Y.detach().cpu().numpy())
            curr_batch = (predictions, images, Y, gt_text)
            write_to_board(writer, collector, epoch, val_dataset, curr_batch, stage='val')
            cofga_v0.train()
        torch.save({ 'epoch': epoch + 1, 'state_dict': cofga_v0.state_dict(),
            'optimizer' : optimizer.state_dict(), }, filename)
    print('Finished ')
if __name__=='__main__':
    main()
