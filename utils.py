import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms, models
import os
import torch.nn.functional as F
from model import *
from Resnet_CCBC import *
from dataset import *

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG_V2(net, 512, 200)

    return net

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

def test(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    _, test_root, _, test_pd, _ = config('bird')
    testset = Dataset(test_root, test_pd, train=False, transform = transform_test, num_positive=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, _, _, _ = net(inputs)
        outputs_com = output_1 + output_2 + output_3

        loss = criterion(outputs_com, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_loss

