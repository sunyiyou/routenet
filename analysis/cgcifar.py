from __future__ import print_function

import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from util.common import progress_bar
from models.resnet import resnet18_cifar, resnet18_fc_ma_cifar, resnet18_fc_ca_cifar
from models.route import CG


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', '-d', default='cifar', type=str, help='dataset')
parser.add_argument('--arch', default='resnet18', type=str, help='arch')
parser.add_argument('--topk', default=1, type=int, help='topk')

args = parser.parse_args()

from models.lenet import ResLeNet
net = resnet18_fc_ma_cifar()
net.fc.weight.data.abs_()

print(net)
print('==> Building model..')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
test_batch_size = 128
args.lr = args.lr * batch_size / 128
# Data
print('==> Preparing data..')
if args.dataset == 'cifar':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    in_channels = 3

elif args.dataset == 'mnist':
    transform_train = transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    transform_test = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    in_channels=1

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
checkpoint = torch.load('./checkpoint/ckpt.t7')
net.load_state_dict(checkpoint['net'])
# start_epoch = 7
# best_acc = checkpoint['acc']
net_p1 = nn.Sequential(*list(net.modules())[:-1])
net_p2 = nn.Sequential(list(net.modules())[-1])
from models.route import RouteFcMaxAct
# new_net_p2 = RouteFcMaxAct(512, 10, True, topk=args.topk).to(device)
new_net_p2 = nn.Linear(512, 10, True).to(device)
new_net_p2.weight.data = torch.abs(new_net_p2.weight.data)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(net_p1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer2 = CG(new_net_p2.parameters(), lr=args.lr, K=args.topk)
optimizer2 = optim.SGD(new_net_p2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Trainingnn.Sequential(list(net.modules())[:-1])
def train(epoch):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        optimizer2.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()


        optimizer.step()
        optimizer2.step(target=targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def train_fc(epoch, feats):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_inds = np.arange(len(trainloader.dataset))#np.random.choice(len(trainloader.dataset), len(trainloader.dataset), replace=False)
    for batch_idx in range(len(batch_inds) // batch_size):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(trainloader.dataset))
        inds = batch_inds[start_idx:end_idx]
        targets = torch.LongTensor(np.array(trainloader.dataset.targets)[inds]).to(device)
        inputs = torch.Tensor(feats[inds]).to(device)

        # if batch_idx % 128//batch_size == 0:
        optimizer2.zero_grad()
        outputs = new_net_p2(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # if batch_idx % 128//batch_size == 0:
        optimizer2.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % (40*128//batch_size) == 0:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test_fc(epoch, feats):
    print("VAL")
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_inds = np.random.choice(len(testloader.dataset), len(testloader.dataset), replace=False)
    with torch.no_grad():
        for batch_idx in range(len(batch_inds) // batch_size):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(testloader.dataset))
            inds = batch_inds[start_idx:end_idx]
            targets = torch.LongTensor(np.array(testloader.dataset.targets)[inds]).to(device)
            inputs = torch.Tensor(feats[inds]).to(device)

            outputs = new_net_p2(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % (40*128//batch_size) == 0:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    print("Acc: %.3f" % acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc



features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.squeeze().cpu())

def extract_features(file, loader):
    if os.path.exists(file):
        return torch.load(file)

    train_loss = 0
    correct = 0
    total = 0
    feats = torch.zeros(len(loader.dataset), 512)
    hook = net.avgpool.register_forward_hook(hook_feature)

    for batch_idx, (inputs, targets) in enumerate(loader):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(loader.dataset))
        inputs, targets = inputs.to(device), targets.to(device)
        del features_blobs[:]
        outputs = net(inputs)
        feats[start_idx:end_idx, :] = features_blobs[0]
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    torch.save(feats, file)
    hook.remove()
    return feats


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 40 == 0:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("Acc: %.3f" % acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

file = 'tmp/ma_t{}_feat_train.pth'.format(args.topk)
net.train()
feats_train = extract_features(file, trainloader)
file = 'tmp/ma_t{}_feat_test.pth'.format(args.topk)
net.eval()
feats_test = extract_features(file, testloader)

test(0)
for epoch in range(start_epoch, start_epoch+200):
    train_fc(epoch, feats_train)
    test_fc(epoch, feats_test)
    adjust_learning_rate(optimizer, epoch)
    if epoch >= 100:
        print(best_acc)
# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)
#     adjust_learning_rate(optimizer, epoch)
