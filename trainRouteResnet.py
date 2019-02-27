
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import argparse

import torchvision.models as models
import easydict as edict
from util.common import *
from loader.data_loader import places365_imagenet_loader
# from models.alexnet import AlexNet, CapAlexNet1, CapAlexNet2

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--arch', default='capalex1', type=str, help='arch')
parser.add_argument('--dataset', default='places365', type=str, help='arch')

args = parser.parse_args()


settings = edict.EasyDict({
    "GPU" : True,
    "IMG_SIZE" : 227,
    "CNN_MODEL" : MODEL_DICT[args.arch],
    "DATASET" : args.dataset,
    "DATASET_PATH" : DATASET_PATH[args.dataset],
    "MODEL_FILE" : 'zoo/alexnet_places365.pth.tar',
    "WORKERS" : 16,
    "BATCH_SIZE" : 256,
    "PRINT_FEQ" : 10,
    "LR" : 0.1,
    "EPOCHS" : 90,
})
settings.OUTPUT_FOLDER = "result/pytorch_{}_{}".format(args.arch, args.dataset)

snapshot_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'snapshot'))
log_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'log'))
print = log_f(os.path.join(log_dir, "%s.txt" % time.strftime("%Y-%m-%d-%H:%M")))


def train_caffenet(model, train_loader, val_loader, dir=None):


    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), settings.LR, momentum=0.9, weight_decay=1e-4)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = settings.LR * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(settings.EPOCHS):

        print('Epoch[%d/%d]' % (epoch, settings.EPOCHS))
        # train
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()


        for i, (input, target) in enumerate(train_loader):

            # input = torch.FloatTensor(input)
            if settings.GPU:
                target = target.cuda()
                input = input.cuda()

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % settings.PRINT_FEQ == 0:
                print('Train: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    top1=top1, top5=top5))
        print(' * TRAIN Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


        # val
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        model.eval()
        for i, (input, target) in enumerate(val_loader):
            start_idx = i * settings.BATCH_SIZE
            end_idx = min((i + 1) * settings.BATCH_SIZE, len(val_loader.dataset))
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
                    i, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    top1=top1, top5=top5))


        print(' * VAL Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        torch.save({
            'epoch': epoch,
            'best_prec1': top1.avg,
            'state_dict': model.state_dict(),
        }, os.path.join(dir, 'epoch_%d.pth' % epoch))

        adjust_learning_rate(optimizer, epoch)


def test_caffenet(test_loader, val_loader, snapshot_dir):
    # model = AdditiveAlexNet(dropout=False)
    model = torch.load(settings.MODEL_FILE)
    if settings.GPU:
        model.cuda()
    model.eval()
    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(settings.EPOCHS):
        # filename = os.path.join(snapshot_dir, 'epoch_%d.pth' % epoch)
        # dt = torch.load(filename)
        # model.load_state_dict(dt['state_dict'])
        # model.load_state_dict(torch.load(settings.MODEL_FILE))

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (input, target) in enumerate(test_loader):
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
        test_acc = top1.avg

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
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
        val_acc = top1.avg

        print('EPOCH [%d] (dropout) VAL Prec@1 %.3f \t TEST Prec@1 %.3f' % (epoch, dt['best_prec1'], test_acc))
        print('EPOCH [%d] VAL Prec@1 %.3f \t TEST Prec@1 %.3f' % (epoch, val_acc, test_acc))




def main():
    # layer = 'pool5'

    val_loader = places365_imagenet_loader(settings, 'val')
    train_loader = places365_imagenet_loader(settings, 'train', shuffle=True, data_augment=True)
    # test_loader = places365_imagenet_loader('test200')


    # model = finetune_model
    model = settings.CNN_MODEL()
    if settings.GPU:
        model.cuda()
    model.train()
    train_caffenet(model, train_loader, val_loader, snapshot_dir)
    # test_caffenet(val_loader, val_loader, snapshot_dir)



if __name__ == '__main__':
    main()