#!/usr/bin/env python3
import argparse
import os,sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataset import MyDataset
import torchvision
from torchvision import datasets , models , transforms 
from torchvision.models import densenet121
from torch.nn import functional as F
from model import densenet121_cls

os.environ['CUDA_VISIBLE_DEVICES']='1'

def arg_parse():
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('-w', '--workers', default=8, type=int, metavar='W',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='B', help='batch size')
    parser.add_argument('--learning_rate', default=1e-2, type=int)
    
    parser.add_argument('--dataset_size', default=50000, type=int, help='training dataset size')
    
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    parser.add_argument('--save_path', default='/home/renyz/CheXpert/model/', type=str, help='path to save checkpoint')
    
    parser.add_argument('--crop_size', dest='crop_size',default=224, type=int, 
                        help='crop size')
    parser.add_argument('--scale_size', dest = 'scale_size',default=256, type=int, # 448
                        help='the size of the rescale image')
    parser.add_argument('--class_num', dest='class_num', type=int, default=3,
                         metavar='class')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=30,
                        help='the number of training epochs')    
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()

    # Create dataloader
    print("==> Creating dataloader...")
    train_list = '/media/renyz/data8g/CheXpert/CheXpert-v1.0-small/total_train_list.txt'   
    test_list = '/media/renyz/data8g/CheXpert/CheXpert-v1.0-small/total_valid_list.txt'

    data_dir = '/media/renyz/data8g/CheXpert/'
    train_loader = get_train_set(data_dir, train_list, args)
    test_loader = get_test_set(data_dir, test_list, args)

    # load the network
    print("==> Loading the network ...")
    model = densenet121(pretrained=False)
    model.classifier = densenet121_cls(num_classes = args.class_num)
    
    # load my model
    #args.resume = '/home/renyz/CheXpert/model/vgg16_epoch_0.pkl'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

    model.cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft , step_size = 7, gamma = 0.1)
    
    #for evaluate
    #validate(test_loader, model, 0, args)
   
    #start 
    model_new = train_model(model, train_loader, test_loader, args, model , criterion, optimizer_ft , exp_lr_scheduler , num_epochs = args.num_epochs)
    print('finish!')


def train_model(checkpoint, train_loader_bbox, test_loader_bbox, args, model , criterion , optimizer , scheduler , num_epochs):
    model.train()
    since = time.time()
    best_model_wts = model.state_dict()  #Returns a dictionary containing a whole state of the module.
    best_acc = 0.0
    top1 = AverageMeter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch , num_epochs - 1))
        print('-' * 10)
        #set the mode of model
        scheduler.step()  #about lr and gamma
        model.train()  #set model to training mode

        running_loss = 0.0
        #Iterate over data
        for i, (input, target) in enumerate(train_loader_bbox):
            target_ori = target.cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            optimizer.zero_grad()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            if (i%20) == 0: 
                print('step: {} totalloss: {loss:.3f}'.format(i, loss = loss))

            loss.backward()  #backward of gradient
            optimizer.step()  #strategy to drop
            
            #print(loss.data.item())
            running_loss += loss.data.item()
            prec, _ = accuracy(output.data,target_ori,topk = (1,2))
            top1.update(prec, input.size(0))
            #break

        epoch_loss = running_loss / (args.dataset_size / args.batch_size)
        print(' Epoch over  Loss: {:.5f}'.format(epoch_loss))
        print(' train Prec@1 {top1.avg:.3f}'.format(top1=top1))

        #save
        if epoch % 1 == 0:
            save_path = args.save_path
            torch.save(model.state_dict(), save_path+'densenet121_epoch_'+str(epoch)+'.pkl')
            print('model already save!')
            validate(test_loader_bbox, model, epoch, args)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))

    return model


def validate(loader, model, epoch, args):
    print('begin validate!')

    model.eval() 
    total_label = []
    start_test = True
    for i, (input, target) in enumerate(loader):
        #print('test step:', i)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        ori_out = model(input_var)

        if start_test:
           total_output_ori = ori_out.data.float()
           total_label = target.data.float()
           start_test = False
        else:
           total_output_ori = torch.cat((total_output_ori, ori_out.data.float()) , 0)
           total_label = torch.cat((total_label , target.data.float()) , 0)
        #break

    _,predict_ori = torch.max(total_output_ori,1)
    acc = torch.sum(torch.squeeze(predict_ori).float() == total_label).item() / float(total_label.size()[0])
    print('Test:')
    print(' *Prec@ ' + str(acc))

    return acc

def get_train_set(data_dir,test_list,args):
    # Data loading code
    # normalize for different pretrain model:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    # center crop
    test_data_transform = transforms.Compose([
          transforms.Resize((scale_size,scale_size)),
          #transforms.CenterCrop(crop_size),
          transforms.RandomCrop(crop_size),
          transforms.ToTensor(),
          normalize,
      ])

    test_set = MyDataset(data_dir, test_list, test_data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,batch_size=args.batch_size, shuffle=True)

    return test_loader


def get_test_set(data_dir,test_list,args):
    # Data loading code
    # normalize for different pretrain model:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    # center crop
    test_data_transform = transforms.Compose([
          transforms.Resize((scale_size,scale_size)),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          normalize,
      ])

    test_set = MyDataset(data_dir, test_list, test_data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,batch_size=args.batch_size, shuffle=False)


    return test_loader

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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=="__main__":
    main()
