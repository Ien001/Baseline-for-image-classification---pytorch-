import csv

label_path = '/media/renyz/data8g/MURA-v1.1/train_image_paths.csv' 
new_train_path = '/media/renyz/data8g/CheXpert/CheXpert-v1.0-small/total_train_list.txt' 
new_test_path = '/media/renyz/data8g/CheXpert/CheXpert-v1.0-small/total_valid_list.txt' 

out_train = open(new_train_path,'a')
out_test = open(new_test_path,'a')


with open(label_path, 'r') as f:
            i = 0
            for l in f.readlines():
                i += 1
                if i <= 20000:
                    #ls = list(l.split(','))   
                    print(l.replace('\n','')+',2')        
                    out_train.write(l.replace('\n','')+',2\n')
                elif i < 35000:

                    #ls = list(l.split(','))
                    print(l.replace('\n','')+',2')
                    out_test.write(l.replace('\n','')+',2\n')
                elif i == 35001:
                    out_train.close()
                    out_test.close()
                    exit(0)
