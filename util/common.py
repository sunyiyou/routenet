import time
import torch
import os
import torch.nn as nn


from models.alexnet import AlexNet
from models.resnet import *

DATASET_PATH = {
    'places365': '/home/sunyiyou/dataset/places365_standard',
    'imagenet': '',
}

NUM_CLASSES = {
    'places365': 365,
    'imagenet': 1000,
}

MODEL_DICT = {
    'alexnet': AlexNet,
    'resnet18': resnet18,
    'resnet18_fc_ma': resnet18_fc_ma,
}

p = print

def log_f(f, console=True):
    def log(msg):
        f.write(msg)
        f.write('\n')
        if console:
            p(msg)
    return log

def dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path















class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def val(settings, loader, model):
    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    # val
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.eval()
    for i, (input, target) in enumerate(loader):
        start_idx = i * settings.BATCH_SIZE
        end_idx = min((i + 1) * settings.BATCH_SIZE, len(loader.dataset))
        # input = torch.FloatTensor(input)
        if settings.GPU:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        fc_output = model(input_var)

        loss = criterion(fc_output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(fc_output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if i % settings.PRINT_FEQ == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1, top5=top5))


    print(' * VAL Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
