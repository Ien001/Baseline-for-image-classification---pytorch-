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
from dataset_pred import MyDataset
import torchvision
from torchvision import datasets , models , transforms 
from torchvision.models import densenet121
from torch.nn import functional as F
from model import densenet121_cls
import shutil
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='1'


def arg_parse():
    parser = argparse.ArgumentParser(description='Torch')
    parser.add_argument('date', default='20190226', type=str)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='batch size')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--crop_size', dest='crop_size',default=224, type=int, #448
                        help='crop size')
    parser.add_argument('--scale_size', dest = 'scale_size',default=448, type=int,
                        help='the size of the rescale image')
   
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()

    # read img path
    make_label_list(args)
    print("==> make_label_list OK...")

    # Create dataloader
    print("==> Creating dataloader...")
    test_list = '/media/renyz/data8g/4Rogen/label_name_'+args.date+'.txt'

    data_dir = ''
    test_loader = get_test_set(data_dir, test_list, args)

    # load the network
    print("==> Loading the network ...")
    model = densenet121(pretrained=False)
    model.classifier = densenet121_cls(num_classes = 3)

    # load my model
    args.resume = '/home/renyz/CheXpert/model/densenet121-0215.pkl'
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
    #for predict
    prediction = predict(test_loader, model, 0, args)
    print('finish predict')

    if not os.path.isdir('/media/renyz/data8g/4Rogen/tmp/'+args.date+'/'):
        os.makedirs('/media/renyz/data8g/4Rogen/tmp/'+args.date+'/')

    total_label_file = '/media/renyz/data8g/4Rogen/tmp/'+args.date+'/label_'+args.date+'.txt'
    total_label_file_obj = open(total_label_file,'w')

    count = 0
    with open(test_list, 'r') as f:
        for l in f.readlines():
            total_label_file_obj.write(l.replace('\n','')+str(int(float(prediction[count])))+'\n')
            count = count + 1

    total_label_file_obj.close()
    print('finish make label!')
    
    move(args)

def predict(loader, model, epoch, args):
    print('begin predict!')
    model.eval() 
    start_test = True
    for i, input in enumerate(loader):
        if (i%20) == 0:
            print('batch processing:',i)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        # compute output
        ori_out = model(input_var)

        if start_test:
           total_output_ori = ori_out.data.float()
           start_test = False
        else:
           total_output_ori = torch.cat((total_output_ori, ori_out.data.float()) , 0)
        #break

    _,predict_ori = torch.max(total_output_ori,1)
    #np.savetxt('/media/renyz/data8g/4Rogen/tmp/label_pred_'+args.date+'.txt', torch.squeeze(predict_ori).float().cpu().numpy())
    #print('ok')

    return torch.squeeze(predict_ori).float().cpu().numpy()


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


def make_label_list(args):
    img_path =  '/media/renyz/data8g/4Rogen/CHEST-DX-'+args.date+'/'
    floder_list = os.listdir(img_path)

    label_path = '/media/renyz/data8g/4Rogen/label_name_'+args.date+'.txt'
    fobj_label = open(label_path, 'w')

    for flodername in floder_list:
        for img_name in os.listdir(img_path+flodername):
            #print(img_path+flodername+'/'+img_name)
            fobj_label.write(img_path+flodername+'/'+img_name+', \n')

    fobj_label.close()
    
    return 0


def resize(img_path, dest_path):
    #img_path = '/media/renyz/data8g/4Rogen/tmp/20190223/filter.1/'
    floder_list = os.listdir(img_path)
    #dest_path = '/media/renyz/data8g/4Rogen/tmp/20190223/filter.1(resized)/'
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    for img_name in floder_list:
        print('resize image name:',img_name)
        img_obj = cv2.imread(img_path+img_name)
        h , w = img_obj.shape[:2]
        res = cv2.resize(img_obj,(int(w*0.2),int(h*0.2)))
        cv2.imwrite(dest_path+img_name,res)

    return 0

def move(args):
    img_path =  '/media/renyz/data8g/4Rogen/CHEST-DX-'+args.date+'/'
    floder_list = os.listdir(img_path)

    dest_path = '/media/renyz/data8g/4Rogen/tmp/'+args.date+'/'

    label_path = dest_path + 'label_'+args.date+'.txt'
    fobj_label = open(label_path, 'r')

    if not os.path.isdir(dest_path+'filter.0/'):
        os.makedirs(dest_path+'filter.0/')
    if not os.path.isdir(dest_path+'filter.1/'):
        os.makedirs(dest_path+'filter.1/')
    if not os.path.isdir(dest_path+'filter.2/'):
        os.makedirs(dest_path+'filter.2/')

    for line in fobj_label.readlines():
        img_name = line.split('/')[-1].split(',')[0]
        label_num = line.split(' ')[1].replace('\n','')
        print('move imgname:',img_name)
        if float(label_num) == 0:
            shutil.copyfile(line.split(',')[0],dest_path+'filter.0/'+img_name)
        elif float(label_num) == 1:
            shutil.copyfile(line.split(',')[0],dest_path+'filter.1/'+img_name)
        elif float(label_num) == 2:
            shutil.copyfile(line.split(',')[0],dest_path+'filter.2/'+img_name)

    fobj_label.close()
    print('finish move!')


    resize(dest_path+'filter.1/', dest_path+'filter.1(resized)/')
    print('finish resize!')


if __name__=="__main__":
    main()
