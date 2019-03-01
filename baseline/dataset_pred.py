#!/usr/bin/env python3
import torch.utils.data as data
from os.path import join
from PIL import Image

class MyDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform = None):
        super(MyDataset, self).__init__()
        print(list_path)

        name_list = []
        
        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename = l.split(',')[0]
                if imagename != 'Path':
                        name_list.append(imagename)


        self.image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform


    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input) 

        return input

    def __len__(self):
        return len(self.image_filenames)
