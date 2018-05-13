import numpy as np
import os
import cv2

class crop_bunch:
    def __init__(self, image, crop_size=80):
        self.image = image
        self.crops_ = dict()
        self.scores = dict()
        self.csize = crop_size

    def take_crops(self):
        # here we do crops
        # save it in the manner of dictionary,
        # key is [up_left_corner_x_y,  low_right_corner_x_y]
        # value is the cropped pieces of image

        img_size = self.image.shape
        height = img_size[0]
        width = img_size[1]

        halfsize = int(self.csize / 2)
        h_n = int(height // halfsize)
        w_n = int(width //  halfsize)

        crop_count = 0
        pos_list = list()
        for i in range(h_n - 1):
            for j in range(w_n - 1):
                pos_list.append((i * halfsize, (i+2) * halfsize, j * halfsize,  (j+2) * halfsize))
                crop_count += 1
            pos_list.append((i * halfsize, (i+2) * halfsize, width-2*halfsize,  width))
            crop_count += 1
            
        for j in range(w_n - 1):
            pos_list.append((height-2*halfsize, height, j * halfsize,  (j+2) * halfsize))
            crop_count += 1

        pos_list.append((height - 2*halfsize, height, width-2*halfsize,  width))
        crop_count += 1

        for pos in pos_list:
            #print(pos)
            crop_area = self.image[pos[0]:pos[1], pos[2]:pos[3], :]
            self.crops_[pos] = crop_area

        cv2.imshow("image", crop_area / 255.0)
        #cv2.imwrite('result.png', crop_area)
        #print(crop_area.shape)
        cv2.waitKey()
        
        return crop_count


def image_to_crop(file_dir):
    #this function should take in directory of the test images,
    #get the images and get sliding window crops out of them
    #e.g. if the size of crop is h x w, then the sliding step should be h/2 and w/2
    #this gives us much overlapping among crops
    #so that we can do some inference over regions from all crops covering them

    # output "crops" should be a list of objects "crops"
    # each object corresponds to one original image
    # containing all the crops we got from that image
    crops_list = list()
    file_list = os.listdir(file_dir)
    for file_name in file_list:
        img_ = cv2.imread(file_dir + '/' + file_name)
        print(img_.shape)

        img_crops_ = crop_bunch(img_)
        crop_total_n = img_crops_.take_crops()
        print(crop_total_n)
        
        #print(img_crops_.crops_.keys())

        crops_list.append(img_crops_)
    
    return crops_list


def make_classification(crop_bunch):
    # take in a list of crops which come from one image
    # give them to classifier we have, get back the score of them belonging to each class

    # ndarray class_n x crop_number
    scores_from_classifier = [0.5, 0.5]
    crop_bunch.scores = scores_from_classifier

    return


def make_inference(crop_bunch):
    # take one object,, and make inference about the image based on the scores
    # expected result would be bounding box showing the suspect region of damage
    # and if possible the degree of suspect for this image to have damage in it.
    inference_result = 0

    return inference_result
    

if __name__ == '__main__':

    image_dir = './test_images'

    crop_bunch_list = image_to_crop(image_dir)

    for crop_bunch_i in crop_bunch_list:
        make_classification(crop_bunch_i)

        infer_result = make_inference(crop_bunch_i)

        # use information in infer_result to do something,
        # e.g. drawing the bounding box on the image and
        # save the new image into another directory
        # showing a notice to the user about the image name and degree of suspect or something

    print("all images have been investigated!")
