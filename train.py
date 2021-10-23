from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import defaultdict

from utils import *
from folder import *
from combine_sampler import *
from dataset import *
            
            
def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    device = torch.device("cuda")

    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_root, _, train_pd, _, _ = config('bird')
    trainset = Dataset(train_root, train_pd, train=True, transform = transform_train, num_positive=1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50', pretrain=True, require_grad=True)
    netp = torch.nn.DataParallel(net)

    if use_cuda:
        net.to(device)
        # cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss(reduce=True, size_average=True)
    optimizer = optim.SGD([
        {'params': net.conv_block1.parameters(), 'lr': 0.005},
        {'params': net.classifier1.parameters(), 'lr': 0.005},
        {'params': net.conv_block2.parameters(), 'lr': 0.005},
        {'params': net.classifier2.parameters(), 'lr': 0.005},
        {'params': net.classifier3.parameters(), 'lr': 0.005},
        {'params': net.conv_block3.parameters(), 'lr': 0.005},
        {'params': net.features.parameters(), 'lr': 0.0005}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, img_pair, targets) in enumerate(trainloader):
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets, img_pair = inputs.to(device), targets.to(device), img_pair[0].to(device)
            inputs, targets, img_pair = Variable(inputs), Variable(targets), Variable(img_pair)
            targets = torch.cat((targets, targets), 0)
            inputs = torch.cat((inputs, img_pair), 0)

            cosine_schedule = False
            lr = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0005]
            for nlr in range(len(optimizer.param_groups)):
                if cosine_schedule:
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
                else:
                    if epoch > 100:
                        optimizer.param_groups[nlr]['lr'] = lr[nlr] / 10
                    elif epoch > 150:
                        optimizer.param_groups[nlr]['lr'] = lr[nlr] / 100
    
            output_1, output_2, output_3, f1, f2, f3 = netp(inputs, block=[8, 8, 0, 0])
            loss1_1 = CELoss(output_1, targets)
            loss1_2 = MSELoss(f1[: batch_size], f1[batch_size:])
            w = loss1_1.item() / loss1_2.item()
            loss1 = loss1_1 + loss1_2 * w * 0.01
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            
            output_1, output_2, output_3, f1, f2, f3 = netp(inputs, block=[4, 4, 4, 0])
            loss2_1 = CELoss(output_2, targets)
            loss2_2 = MSELoss(f2[: batch_size], f2[batch_size:])
            w = loss2_1.item() / loss2_2.item()
            loss2 = loss2_1 + loss2_2 * w * 0.05
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            
            output_1, output_2, output_3, f1, f2, f3 = netp(inputs, block=[2, 2, 2, 2])
            loss3_1 = CELoss(output_3, targets)
            loss3_2 = MSELoss(f3[: batch_size], f3[batch_size:])
            w = loss3_1.item() / loss3_2.item()
            loss3 = loss3_1 + loss3_2 * w * 0.1
            optimizer.zero_grad()
            loss3.backward()
            optimizer.step()
            
            _, predicted = torch.max(output_3.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (batch_idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1), train_loss3 / (batch_idx + 1)))

        conduct_test = True
        if conduct_test:
            val_acc, val_loss = test(net, CELoss, batch_size // 4)
            if val_acc >= max_val_acc:
                max_val_acc = val_acc
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_loss))
       
        net.cpu()
        torch.save(net, './' + store_name + '/last.pth')
        net.to(device)


train(nb_epoch=200,
         batch_size=16,
         store_name='cub-200-2011',
         resume=False,
         start_epoch=0,
         model_path='')
