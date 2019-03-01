import csv
import os
import sys

img_path =  '/media/renyz/data8g/4Rogen/CHEST-DX-20190218/'
floder_list = os.listdir(img_path)

label_path = '/media/renyz/data8g/4Rogen/label_name_20190218.txt'
fobj_label = open(label_path, 'w')

for flodername in floder_list:
    for img_name in os.listdir(img_path+flodername):
        print(img_path+flodername+'/'+img_name)
        fobj_label.write(img_path+flodername+'/'+img_name+', \n')

fobj_label.close()

