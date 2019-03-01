#!/usr/bin/env python3
import torch.utils.data as data
from os.path import join
from PIL import Image

class MyDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform = None):
        super(MyDataset, self).__init__()
        print(list_path)

        name_list = []
        label_list = []
        
        with open(list_path, 'r') as f:
            for l in f.readlines():
                #print(l.split(','))
                imagename = l.split(',')[0]
                if imagename != 'Path':
                        name_list.append(imagename)
                                # other
                if l.split(',')[1].replace('\n','') == '2':
                        label_list.append(2)                
                # zheng
                elif l.split(',')[4] == 'AP' or l.split(',')[4] == 'PA' :
                        label_list.append(1)                
                # ce
                else:
                        label_list.append(0)

        self.image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform
        self.label_list = label_list
        

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        #print(imagename)

        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input) 

        target = self.label_list[index]
        #print(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
