import cv2
import numpy as np
from os import listdir
from os.path import splitext
import sys


def find_images(file_or_dir):
	#print("name is " + file_or_dir)
	if not (file_or_dir.endswith('.png') or file_or_dir.endswith('.jpg')):
		# we assume it is a directory
		name_list = listdir(file_or_dir)
		for name_i in name_list:
			find_images(file_or_dir + '/' + name_i)
	else:
		# we assume it must be an image
		#print(file_or_dir)
		img = cv2.imread(file_or_dir)
		#print(img.shape)
		gray = image_process(img)
		filename, file_extension = splitext(file_or_dir)
		outfile_ = filename + '_gray' + file_extension
		cv2.imwrite(outfile_, gray)
		

def image_process(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
	
	####################################################################
	# you need to input the path for which you want to have all images #
	# in it or its subfolders converted from rgb to gray               #
	# all gray images are stored just beside the original with         #
	# name as "original name_gray.original extension"                  #
	# example: python intoblack './test'                               #
	# do not include '/' after the final directory.                    #
	####################################################################
	
	if len(sys.argv) >= 2:
		path_name = list()
		for i in range(1, len(sys.argv)):
			path_name.append(sys.argv[i])
	else:
		print("hey, please at least put in one path!")
		print("open the script and see how to use it")

	for path_i in path_name:
		find_images(path_i)
		
