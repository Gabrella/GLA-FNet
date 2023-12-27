#import
import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import datetime
import patch_loader as patchloader
from models import resnet_binary as model_ldl
from models import model_cluster_upload as GLA
from helper import adjust_learning_rate, accuracy, save_checkpoint, AverageMeter, ProgressMeter, RecorderMeter

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='./datasets/RAFDB/')
parser.add_argument('--checkpoint_path', type=str, default='./1v_checkpoint/' + time_str + 'model.pth.tar')
parser.add_argument('--best_checkpoint_path', type=str, default='./1v_checkpoint/'+time_str+'model_best.pth.tar')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to runs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--gamma', default=0.1, type=float, metavar='gamma')
parser.add_argument('--af', '--adjust-freq', default=30, type=int, metavar='N', help='adjust learning rate frequency')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('-f', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('-feature_dim', type=int, default=24)
parser.add_argument('-class_num', type=int, default=7)
parser.add_argument('--gpu', type=str, default='6')
args = parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print('Training Time: ' + now.strftime("%m-%d %H:%M"))

    models = GLA.resnet_face()
    models.fc = nn.Linear(2048, 12666)
    models = torch.nn.DataParallel(models).cuda()
    # pretrained parameters
    checkpoint = torch.load('./checkpoint/resnet50_pretrained_on_msceleb.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    model_dict = models.state_dict()
    pretrained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    models.load_state_dict(model_dict)
    models.module.fc = nn.Linear(args.feature_dim, args.class_num).cuda()
    
    # Binary_FRG
    model_ldg = model_ldl.resnet50()
    model_ldg = torch.nn.DataParallel(model_ldg).cuda()
    model_checkpoint = torch.load('./binary_pretrain/[12-05]-[00-14]-model_best.pth')
    model_dict = model_ldg.state_dict()
    model_dict.update(model_checkpoint)
    model_ldg.load_state_dict(model_dict)

    # loss function and optimizer
    criterion_train = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'distribution': distance
    }
    criterion_val = nn.CrossEntropyLoss().cuda()
    optimizer = {
        'softmax': torch.optim.SGD(models.parameters(),
                                   lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    }
    recorder = RecorderMeter(args.epochs)
    best_acc = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            models.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}".format)

    # data loading
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')

    train_dataset = patchloader.patch_generator(traindir,
                                                transforms.Compose([transforms.ToPILImage(),
                                                           transforms.RandomHorizontalFlip()]))
    
    val_dataset = patchloader.patch_generator(testdir,
                                              transforms.Compose([transforms.ToPILImage()]))
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                             drop_last=True)
    
    if args.evaluate:
        validate(val_loader, models, criterion_val, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = adjust_learning_rate(optimizer['softmax'], epoch, args.lr, args.gamma, args.af)
        print('Current learning rate: ', current_learning_rate)
        txt_name = './1v_log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        train_acc, train_loss = train(train_loader, models, model_ldg, criterion_train, optimizer, epoch, args)

        val_acc, val_loss = validate(val_loader, models, criterion_val, epoch, args)

        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        curve_name = time_str + 'log.png'
        recorder.plot_curve(os.path.join('./1v_log/', curve_name))

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())
        txt_name = './1v_log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'state_dict': models.state_dict(),
                         'epoch': epoch+1,
                         'best_acc': best_acc,
                         'recorder': recorder}, 
                        args.checkpoint_path, args.best_checkpoint_path, is_best)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './1v_log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')

def train(train_loader, model, model_lgd, criterion, optimizer, epoch, args):
    losses = {
        'softmax': AverageMeter('softmax', ':.4f'),
        'distribution': AverageMeter('distribution', ':.4f'),
        'total': AverageMeter('total', ':.4f')
    }
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses['total'], top1],
                             prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    model_lgd.eval()

    for i, (images, eyes, mouth, nose, lcheek, rcheek, target) in enumerate(train_loader):
        images = images.cuda()
        eyes = eyes.cuda()
        mouth = mouth.cuda()
        nose = nose.cuda()
        lcheek = lcheek.cuda()
        rcheek = rcheek.cuda()
        target = target.cuda()

        # output
        output, output_label = model(images, eyes, mouth, nose, lcheek, rcheek)
        # compute loss
        # obtain 
        with torch.no_grad():
            soft_list, soft_label_list = model_lgd(images)
            soft_label = torch.clone(output_label)
            for m in range(0, output.size()[0]):
                soft_label[m] = soft_label_list[target[m]][m]
        loss_total = 0.0
        loss_softmax = criterion['softmax'](output, target)
        loss_distribution = criterion['distribution'](output_label, soft_label)
        lamda = 0.4
        loss_total = loss_softmax + lamda * loss_distribution
        
        acc1, _ = accuracy(output.data, target, topk=(1, 5))
        losses['softmax'].update(loss_softmax.item(), images.size(0))
        # losses['diff'].update(loss_diff.item(), images.size(0))
        losses['distribution'].update(loss_distribution.item(), images.size(0))
        losses['total'].update(loss_total.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer['softmax'].zero_grad()
        loss_total.backward()
        optimizer['softmax'].step()

        if i % args.print_freq == 0:
            progress.display(i, time_str)
        
    return top1.avg, losses['total'].avg


def validate(val_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix="Test: ")
    model.eval()
    with torch.no_grad():
        for i, (images, eyes, mouth, nose, lcheek, rcheek, target) in enumerate(val_loader):
            images = images.cuda()
            eyes = eyes.cuda()
            mouth = mouth.cuda()
            nose = nose.cuda()
            lcheek = lcheek.cuda()
            rcheek = rcheek.cuda()
            target = target.cuda()
            
            output, _ = model(images, eyes, mouth, nose, lcheek, rcheek)
            loss = criterion(output, target)

            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                    progress.display(i, time_str)
        
        print(' *** Accuracy {top1.avg:.3f}  *** '.format(top1=top1))
        with open('./1v_log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg

def cross_entropy(predict_label, true_label):
    return torch.mean(- true_label * torch.log(predict_label))

def distance(input, target):
    return torch.mean(F.pairwise_distance(input, target, p=2)/2)

if __name__ == '__main__':
    main()
