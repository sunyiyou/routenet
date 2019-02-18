# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
from common import *
# Hyper Parameters
num_epochs = 30
learning_rate = 0.1

def train(model, train_loader, val_loader):
    model.train()
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # Train the Model

    for epoch in range(num_epochs):
        print('Epoch[%d/%d]' % (epoch, num_epochs))
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = [Variable(l) for l in labels]
            fc_output = model(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = 0
            for li in range(len(labels)):
                loss += criterion(fc_output[:, 36 * li:36 * (li + 1)], labels[li])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            prec1, prec5 = 0, 0
            for li in range(len(labels)):
                prec1_p, prec5_p = accuracy(fc_output.data[:, 36 * li:36 * (li + 1)], labels[li], topk=(1, 5))
                prec1 += prec1_p
                prec5 += prec5_p
            prec1 /= len(labels)
            prec5 /= len(labels)
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            if i % 10 == 0:
                print('TRAIN: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * TRAIN Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        val(model, val_loader)

        torch.save({
            'epoch': epoch,
            'best_prec1': top1.avg,
            'state_dict': model.state_dict(),
        }, 'model/epoch_%d.pth' % epoch)


def val(model, loader):
    model.eval()
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # Train the Model


    for i, (images, labels) in enumerate(loader):
        images = Variable(images)
        labels = [Variable(l) for l in labels]
        fc_output = model(images)
        # print(predict_labels.type)
        # print(labels.type)
        loss = 0
        for li in range(len(labels)):
            loss += criterion(fc_output[:, 36 * li:36 * (li + 1)], labels[li])

        # measure accuracy and record loss
        prec1, prec5 = 0, 0
        for li in range(len(labels)):
            prec1_p, prec5_p = accuracy(fc_output.data[:, 36 * li:36 * (li + 1)], labels[li], topk=(1, 5))
            prec1 += prec1_p
            prec5 += prec5_p
        prec1 /= len(labels)
        prec5 /= len(labels)
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        if i % 10 == 0:
            print('VAL: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    print(' * VAL Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))



def main():
    model = CNN()
    train_dataloader = my_dataset.get_train_data_loader()
    test_dataloader = my_dataset.get_test_data_loader()
    train(model, train_dataloader, test_dataloader)



if __name__ == '__main__':
    main()


