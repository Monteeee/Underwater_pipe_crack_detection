import numpy as np

class crop_bunch:
    def __init__(self, image):
        self.image = image
        self.crops_ = dict()
        self.scores = dict()

    def take_crops(self):
        # here we do crops
        # save it in the manner of dictionary,
        # key is [up_left_corner_x_y,  low_right_corner_x_y]
        # value is the cropped pieces of image


def image_to_crop(file_dir):
    #this function should take in directory of the test images,
    #get the images and get sliding window crops out of them
    #e.g. if the size of crop is h x w, then the sliding step should be h/2 and w/2
    #this gives us much overlapping among crops
    #so that we can do some inference over regions from all crops covering them

    # output "crops" should be a list of objects "crops"
    # each object corresponds to one original image
    # containing all the crops we got from that image
    return crops_list


def make_classification(crop_bunch):
    # take in a list of crops which come from one image
    # give them to classifier we have, get back the score of them belonging to each class

    # ndarray class_n x crop_number
    all_crops.scores = scores_from_classifier


def make_inference(crop_bunch):
    # take one object,, and make inference about the image based on the scores
    # expected result would be bounding box showing the suspect region of damage
    # and if possible the degree of suspect for this image to have damage in it.

    return inference_result
    

if __name__ == '__main__':

    image_dif = 'xxx'

    crop_bunch_list = image_to_crop(image_dif)

    for crop_bunch_i in crop_bunch_list:
        make_classification(crop_bunch_i)

        infer_result = make_inference(crop_bunch_i)

        # use information in infer_result to do something,
        # e.g. drawing the bounding box on the image and
        # save the new image into another directory
        # showing a notice to the user about the image name and degree of suspect or something

    print("all images have been investigated!")
