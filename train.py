import numpy as np
import cv2
import argparse
import torchvision.models as models
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_loader import MafatDataset
from tensorboardX import SummaryWriter
from PIL import Image
from sklearn.metrics import average_precision_score as MAP

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
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--freeze', dest='freeze', action='store_true', help='freeze params for base model')
parser.add_argument('--weighted-loss', dest='weighted_loss', action='store_true', help='use weights in log loss')
parser.add_argument('--preload', dest='preload', action='store_false', help='preload all images into memory before training')
parser.add_argument('--tag', type=str, default='baseline', help='tag - name of this experiment')
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


def main():
    model = models.resnet18(pretrained=True)
    assert not args.freeze, 'not yet supported'
    assert not args.weighted_loss, 'not yet supported'
    model.fc = torch.nn.Linear(in_features=512, out_features=37, bias=True)
    cofga_v0 = model.cuda()
    #nn.Sequential(model, nn.Sigmoid()).cuda()

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.SGD(cofga_v0.parameters(), lr=args.lr, momentum=args.momentum)

    filename = 'resnet18_BCEloss37%s.checkpoint.pth.tar'%args.tag
    try:
        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    except:
        throw
        start_epoch=0
        print 'failed to load checkoutput'
    train_dataset = MafatDataset('data/train/train.csv', preload=args.preload)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    #val_loader = torch.utils.data.DataLoader(MafatDataset('data/val/train.csv'), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    writer = SummaryWriter('runs/renset18/%s'%args.tag)
    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = None
        running_accuracy = 0.0
        for i, data in enumerate(train_loader):
            images,labels, gt_text = data
            X = Variable(images).cuda()
            outputs = cofga_v0(X)
            Y = Variable(labels).cuda()
            loss = criterion(outputs, Y)
            s=loss.mean()
            optimizer.zero_grad()
            s.backward()
            optimizer.step()

            if running_loss is None:
                running_loss = loss.mean(0)
            else:
                running_loss += loss.mean(0)
            print 'loss - ', loss.data.mean()
            #accuracy, ac3, ac5, train_images_vis = metrics(Y, outputs, images)
            #running_accuracy += accuracy
            #running_ac3 += ac3
            #running_ac5 += ac5
            if i%10==0:
                print 'running loss ', running_loss
                sigmoid = nn.Sigmoid()
                prediction = sigmoid(outputs)
                pred_text = map(train_dataset.labels_to_text, prediction.detach().cpu().numpy()[:16])
                it = epoch*len(train_loader)+i
                writer.add_scalar('Train/loss', running_loss.mean().data/(i+1), it)
                names = train_dataset.get_class_names()

                y_true = Y.cpu().detach().numpy()
                y_score = outputs.cpu().detach().numpy()
                mask = np.count_nonzero(y_true,axis=0)>0
                average_prec = MAP(y_true[:,mask],y_score[:,mask],average=None)
                per_class =  dict(zip(np.array(names)[mask], average_prec))
                print per_class

                for name,l in zip(names, running_loss.data):
                    print name,l
                    writer.add_scalar('Train/loss_%s'%name, l/(i+1), it)
                    if name in per_class:
                        writer.add_scalar('Train/map_%s'%name, per_class[name], it)


                #writer.add_scalar('Train/accuracy', running_accuracy/(i+1), it)
                #writer.add_scalar('Train/accuracy@3', running_ac3/(i+1), it)
                #writer.add_scalar('Train/accuracy@5', running_ac5/(i+1), it)
                print pred_text
                grid=display_images(X[:16], gt_text[:16], pred_text[:16], nrow=4)
                writer.add_image('Train/images', grid, it)
                #val_loss = 0.0
                #val_accuracy = 0.0
                #val_ac3 = 0.0
                #val_ac5 = 0.0
                #cofga_v0.eval()
                #for j, data in enumerate(val_loader):
                #    inputs, labels, images = data
                #    X = Variable(torch.cat([inputs[:,0,:,:], inputs[:,1,:,:]])).cuda()
                #    outputs = cofga_v0(X)
                #    Y = Variable(labels).cuda()
                #    val_loss += criterion(outputs, Y).data
                #    accuracy, ac3, ac5, val_images_vis = metrics(Y, outputs, images)
                #    val_accuracy += accuracy
                #    val_ac3 += ac3
                #    val_ac5 += ac5
                #writer.add_scalar('Val/loss', val_loss/(j+1), it)
                #writer.add_scalar('Val/accuracy', val_accuracy/(j+1), it)
                #writer.add_scalar('Val/accuracy@3', val_ac3/(j+1), it)
                #writer.add_scalar('Val/accuracy@5', val_ac5/(j+1), it)
                #ims = [im for im in itertools.chain(*val_images_vis)]
                #writer.add_image('Val/TOP5', make_grid( ims, nrow=6), it)
                #print('[%d, %d] train, val loss: %.3f, %.3f' % (epoch + 1, i + 1,
                #    running_loss/(i+1), val_loss/(j+1)))
                #print('         train accuracy : %.3f, val accuracy = %.3f' % ( running_accuracy / (i+1), val_accuracy/(j+1)))
                #cofga_v0.train()
            torch.save({ 'epoch': epoch + 1, 'state_dict': cofga_v0.state_dict(),
                'optimizer' : optimizer.state_dict(), }, filename)
    print('Finished ')
if __name__=='__main__':
    main()
