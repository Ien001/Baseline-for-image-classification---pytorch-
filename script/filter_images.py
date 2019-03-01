# coding=utf-8
import csv
import os
import sys
import shutil
import cv2
import argparse

def resize(img_src,img_dest,ratio=0.2):
	img_obj = cv2.imread(img_src)
	h,w=img_obj.shape[:2]
	res=cv2.resize(img_obj,(int(w*ratio),int(h*ratio)))
	cv2.imwrite(img_dest,res)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("dest_path", help="dest path")
	parser.add_argument("label_name", default="label_name.txt",help="label name file")
	#parser.add_argument("label_pred", default="label_pred.txt",help="prediction file")
	args = parser.parse_args()

	dest_path = args.dest_path # '/media/renyz/data8g/4Rogen/tmp/20190223/'

	label_path_name =  args.label_name
	fobj_label_name = open(label_path_name, 'r')

	#label_list = []
	#label_path_label = args.label_pred
	#fobj_label_label = open(label_path_label, 'r')

	#for line in fobj_label_label.readlines():
	#	label_list.append(int(float(line.replace('\n',''))))

	#fobj_label_label.close()
	#l = len(label_list)
	#print('label length:', l)


	if not os.path.isdir(dest_path+'filter.0/'):
		os.makedirs(dest_path+'filter.0/')
	if not os.path.isdir(dest_path+'filter.1/'):
		os.makedirs(dest_path+'filter.1/')
	if not os.path.isdir(dest_path+'filter.2/'):
		os.makedirs(dest_path+'filter.2/')

	i = 0
	for line in fobj_label_name.readlines():
		img_fp = line.split(',')[0]
		img_name = img_fp.split('/')[-1]
		label_num = line.split(' ')[1].replace('\n','')
		img_class = int(label_num)
		#print(label_list[i])
		if img_class == 0:
			#print('0000000000000000000')
			#print("move "+img_fp+" to "+dest_path+'filter.0/'+img_name)
			shutil.move(img_fp,dest_path+'filter.0/'+img_name)
			
		elif img_class == 1:
			#print('1111111111111111111')
			#print("resize "+img_fp+" to "+dest_path+'filter.1/'+img_name)
			resize(img_fp,dest_path+'filter.1/'+img_name)
		elif img_class == 2:
			#print('2222222222222222222')
			#print("move "+img_fp+" to "+dest_path+'filter.2/'+img_name)
			shutil.move(img_fp,dest_path+'filter.2/'+img_name)

		i = i + 1
		if(i%1000 == 0):
			print(i)

	fobj_label_name.close()
	
	print("Processed "+str(i)+" images")