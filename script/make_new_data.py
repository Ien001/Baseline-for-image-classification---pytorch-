import csv

label_path = '/media/renyz/data8g/CheXpert/CheXpert-v1.0-small/train.csv' 
new_train_path = '/media/renyz/data8g/CheXpert/CheXpert-v1.0-small/train_new.txt' 
new_test_path = '/media/renyz/data8g/CheXpert/CheXpert-v1.0-small/valid_new.txt' 

out_train = open(new_train_path,'a')
out_test = open(new_test_path,'a')


with open(label_path, 'r') as f:
            i = 0
            for l in f.readlines():
                i += 1
                if i <= 50000:
                    #ls = list(l.split(','))           
                    out_train.write(l)
                elif i < 75000:
                    print(i)
                    #ls = list(l.split(','))
                    out_test.write(l)
                elif i == 75001:
                    out_train.close()
                    out_test.close()
                    exit(0)
