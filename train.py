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
from loss import RankLoss
from collate import default_collate as collate_fn
from torch.utils.data import WeightedRandomSampler
from mobilenetv2.models.imagenet import mobilenetv2
import pretrainedmodels
import pretrainedmodels.utils as utils


# To control gpus use CUDA_VISIBLE_DEVICES env var


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--nodoublefc', action='store_true')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', action='store_false')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false', help='use pre-trained model')
parser.add_argument('--freeze', dest='freeze', default=0, type=int, help='number of childern to freeze for base model')
parser.add_argument('--weighted-loss', dest='weighted_loss', action='store_true', help='use weights in log loss')
parser.add_argument('--weighted-sample', dest='weighted_sample', action='store_true', help='use weights in log loss')
parser.add_argument('--preload', dest='preload', action='store_false', help='preload all images into memory before training')
parser.add_argument('--tag', type=str, default='baseline', help='tag - name of this experiment')
parser.add_argument('--start-tag', type=str, default=None, help='start-tag - name of experiment use as pretrain')
parser.add_argument('--architect', type=str, default='resnet18', help='architecture to use')
parser.add_argument('--loss', type=str, default=None, help='loss function to use. default BCE ')
parser.add_argument('--no-split', action='store_true', help='Train on the whole train set (do not split train into test-val)' )
parser.add_argument('--context', action='store_true', help='mask the detections. use only context')
parser.add_argument('--normalize_rotation', action='store_true', help='mask the detections. use only context')
parser.add_argument('--normalize_size', action='store_false', help='mask the detections. use only context')
parser.add_argument('--sanitize', action='store_true', help='sanitize output probabilities acording to subclasses' )
parser.add_argument('--image-group-file', default='data/train_image_groups.yaml', action='store_true', help='sanitize output probabilities acording to subclasses' )
args = parser.parse_args()


def display_images(X, text, pred_text, nrow=4):
    images = []
    mean = np.array([0.485, 0.456, 0.406])[:,np.newaxis,np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[:,np.newaxis,np.newaxis]
    for im, t1, t2 in zip(X,text, pred_text):
        im = im.cpu().detach().numpy()
        im = im*std+mean # unnormalize
        I = (np.transpose(im, [1,2,0]).copy()*225).astype(np.uint8)
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

        images.append(torch.from_numpy(np.transpose(I,[2,0,1])))
    return make_grid(images, nrow=4)

def load_model():
    if args.architect.startswith('resnet'):
        if args.architect == 'resnet18':
            model = models.resnet18(pretrained=True)
        if args.architect == 'resnet50':
            model = models.resnet50(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        if args.nodoublefc is True:
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=37)
        else:
            model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features, out_features=1024),
                                     nn.ReLU(), nn.Linear(in_features=1024, out_features=37))  # change dis shit
    elif args.architect == 'mobilenet':
        model = mobilenetv2()
        model.load_state_dict(torch.load('./mobilenetv2/pretrained/mobilenetv2-36f4e720.pth'))
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        if args.nodoublefc is True:
            model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=37)
        else:
            model.classifier = nn.Sequential(nn.Linear(in_features=model.classifier.in_features, out_features=1024),
                                             nn.ReLU(), nn.Linear(in_features=1024, out_features=37))  # change dis shit
    elif args.architect in pretrainedmodels.__dict__: # no blabla fully conv for these models
        model = pretrainedmodels.__dict__[args.architect](num_classes=1000, pretrained='imagenet')
        load_img = utils.LoadImage()
        tf_img = utils.TransformImage(model)
    else:
        raise 'no known architecture %s'%args.architect
    filename = '%s.checkpoint.pth.tar'%(args.tag)
    if args.start_tag is not None:
        start_filename = '%s.checkpoint.pth.tar'%(args.start_tag)
    else:
        start_filename = filename

    checkpoint = None
    if os.path.exists(start_filename) and args.resume:
        checkpoint = torch.load(start_filename)
        model.load_state_dict(checkpoint['state_dict'])
        print 'loaded weights from file ', start_filename
    for i, child in enumerate(model.children()):
        print child
        if i < args.freeze:
            for param in child.parameters():
                param.requires_grad = False
    return model.cuda(), checkpoint, filename

def init_dataset():
    print 'loading Dataset'
    dataset_args={}
    if args.context:
        dataset_args.update(dict(mask_detection=True, boarder_ratio=5, patch_size=224))
    if args.normalize_size:
        dataset_args.update(dict(normalize_size=True))
    if args.normalize_rotation:
        dataset_args.update(dict(normalize_rotation=True))

    if args.no_split:
        train_dataset = MafatDataset('data/train.csv', 'data/answer.csv', 'data/training imagery', args.preload,**dataset_args)
        val_dataset = train_dataset
    else:
        train_dataset, val_dataset = create_train_val_dataset('data/test.csv', 'data/answer.csv', 'data/test imagery',
                image_group_file=args.image_group_file, preload=args.preload, **dataset_args)
    return train_dataset, val_dataset

def write_to_board(writer, collector, it, dataset, curr_batch,  stage='train'):
    """
    it = current iteration to log
    """
    prediction, images, Y, gt_text = curr_batch
    pred_text = map(dataset.labels_to_text, prediction.detach().cpu().numpy()[:16])

    average_prec, names, num_instances = collector.calc_map()
    per_class_map =  dict(zip(np.array(names), average_prec))
    per_class_instance =  dict(zip(np.array(names), num_instances))

    loss = collector.calc_loss()
    writer.add_scalars('Loss', {'total_%s'%stage: loss.mean()}, it)
    writer.add_scalars('Loss', {'%s_%s'%(name,stage):l
                for name,l in zip(names, loss)}, it)
    writer.add_scalars('MAP', {'total_%s'%stage: average_prec.mean()}, it)
    map_per_class = {'%s_%s'%(name,stage):val for name, val in per_class_map.items()}
    writer.add_scalars('MAP', map_per_class , it)
    grid=display_images(images[:16], gt_text[:16], pred_text[:16], nrow=4)
    writer.add_image('%s/images'%stage, grid, it)

    for key,value in sorted(per_class_map.iteritems(), key=lambda (k,v): (v,k)):
        print "MAP_%s %s: %0.2f, # %d " % (stage, key, value, per_class_instance[key])

def evaluate():
    output_file = 'answer_%s.csv'%args.tag
    model, _, _ = load_model()
    model.eval()
    dataset_args={}
    if args.context:
        dataset_args.update(dict(mask_detection=True, boarder_ratio=5, patch_size=224))
    if args.normalize_size:
        dataset_args.update(dict(normalize_size=True))
    if args.normalize_rotation:
        dataset_args.update(dict(normalize_rotation=True))
    dataset = MafatDataset('data/test.csv', 'data/answer.csv', 'data/test imagery', 
            preload=args.preload, augment=False, **dataset_args)
    writer = SummaryWriter('runs/%s/%s'%(args.architect,args.tag))
    train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,collate_fn=collate_fn)
    sigmoid = nn.Sigmoid()
    collector = PredictionCollector(dataset.get_class_names())
    with torch.no_grad():
        for it, data in enumerate(train_loader):
            images,labels, gt_text = data
            X = Variable(images).cuda()
            outputs = model(X)
            prediction = sigmoid(outputs)
            ids = [int(text.split(',')[0]) for text in gt_text]
            collector.add(ids, prediction)

        pred_text = map(dataset.labels_to_text, prediction.detach().cpu().numpy()[:16])
        grid=display_images(X[:16], gt_text[:16], pred_text[:16], nrow=4)
        cv2.imsave(grid.numpy()[0], 'test_grid.png')
        writer.add_image('Test/images', grid, it)
        collector.save(output_file)


def train():
    model, checkpoint, filename = load_model()
    model.train()
    if args.loss == 'RankLoss':
        print 'starting RankLoss'
        criterion = RankLoss()
    else:
        assert args.loss is None
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    sigmoid = nn.Sigmoid()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    start_epoch = 0
    if checkpoint is not None:
        if args.start_tag is None:
            print 'loading optimizer state'
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        epochs = args.epochs+start_epoch

    writer = SummaryWriter('runs/%s/%s'%(args.architect,args.tag))

    train_dataset, val_dataset = init_dataset()
    sampler=None
    if args.weighted_sample:
        print 'calc sample weights'
        weights, num_samples = train_dataset.get_weights()
        print 'init sampler'
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        print 'done init sampler'
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=(sampler is None),
            num_workers=args.workers, collate_fn=collate_fn,sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    #with torch.no_grad():
    #    collector = PredictionCollector(val_dataset.get_class_names())
    #    model.eval()
    #    for _, data in enumerate(val_loader):
    #        images,labels, gt_text = data
    #        X = Variable(images).cuda()
    #        outputs = model(X)
    #        Y = Variable(labels).cuda()
    #        ids = [int(text.split(',')[0]) for text in gt_text]
    #        predictions = sigmoid(outputs)
    #        collector.add(ids, predictions.detach().cpu().numpy(), Y.detach().cpu().numpy())
    #    curr_batch = (predictions, images, Y, gt_text)
    #    write_to_board(writer, collector, start_epoch, val_dataset, curr_batch, stage='val')
    #    model.train()
    for epoch in range(start_epoch+1, epochs):
        collector = PredictionCollector(train_dataset.get_class_names())
        for i, data in enumerate(train_loader):
            images,labels, gt_text = data
            X = Variable(images).cuda()
            outputs = model(X)
            Y = Variable(labels).cuda()
            ids = [int(text.split(',')[0]) for text in gt_text]
            predictions = sigmoid(outputs)
            collector.add(ids, predictions.detach().cpu().numpy(), Y.detach().cpu().numpy())

            loss = criterion(outputs, Y)
            if args.weighted_loss:
                loss = loss.sum(0)/torch.clamp(Y.sum(0),1)
            s = loss.mean()
            optimizer.zero_grad()
            s.backward()
            optimizer.step()
            print 'epoch %d, i %d, loss %0.2f'%(epoch,i, s)

        curr_batch = (predictions, images, Y, gt_text)
        write_to_board(writer, collector, epoch, train_dataset, curr_batch, stage='train')
        torch.save({ 'epoch': epoch + 1, 'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(), }, filename)
        if args.no_split:
            continue
        with torch.no_grad():
            collector = PredictionCollector(val_dataset.get_class_names())
            model.eval()
            for _, data in enumerate(val_loader):
                images,labels, gt_text = data
                X = Variable(images).cuda()
                outputs = model(X)
                Y = Variable(labels).cuda()
                ids = [int(text.split(',')[0]) for text in gt_text]
                predictions = sigmoid(outputs)
                collector.add(ids, predictions.detach().cpu().numpy(), Y.detach().cpu().numpy())
            curr_batch = (predictions, images, Y, gt_text)
            write_to_board(writer, collector, epoch, val_dataset, curr_batch, stage='val')
            model.train()
    print('Finished ')


if __name__=='__main__':
    if args.evaluate:
        evaluate()
    else:
        train()
