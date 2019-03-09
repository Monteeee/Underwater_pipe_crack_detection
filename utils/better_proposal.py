import numpy as np
import cv2
import itertools
import imutils


class BetterProposal:
    """Well, like many other things in this world, methods in this class,
        could make it beautiful, at the cost of making it slow."""
    CORE_SIZE = 20  # "core" is a 20x20 pixels small patch
    PATCH_SIZE = 80  # we have 80x80 pixels input images for the network

    @staticmethod
    def better_sliding_window(image, stepsize, windowsize):
        """this function serves demo by providing better sliding window.
        It should return a generator, that can be iterated over image
        and provide (x, y) which is the right up corner of window and
        window image itself"""

        img_size = image.shape

        for y in range(0, img_size[0], stepsize):
            for x in range(0, img_size[1], stepsize):
                yield (x, y, image[y:y+windowsize[1], x:x + windowsize[0]])

    @staticmethod
    def better_proposal(pos_score_group):
        """
        this function serves demo by providing better proposals.
        It should receive a group of position-score pair, r
        epresenting raw proposals for detection.
        Then this function will find better
        proposals by summarizing raw proposals
        """

        better_prop = list()
        return better_prop

    @classmethod
    def patches_around_core(cls, core_pos, image):
        """t
        his function finds all legal patches around core
        (a small group pixels, one or several)
        it returns the cropped patches in required shape.
        """
        patches = list()

        image_size = image.shape
        patch_size = (cls.PATCH_SIZE, cls.PATCH_SIZE)

        core_corners = np.array([[core_pos[0], core_pos[1]],
                                 [core_pos[0] + cls.CORE_SIZE, core_pos[1]],
                                 [core_pos[0], core_pos[1] + cls.CORE_SIZE],
                                 [core_pos[0] + cls.CORE_SIZE, core_pos[1] +
                                 cls.CORE_SIZE]])

        # get four simple ones (no rotation)
        # half_psize = int(cls.PATCH_SIZE/2)
        for corner in core_corners:
            patch = cv2.getRectSubPix(image, patch_size, tuple(corner))
            patches.append(patch)

        # filter out patches that are out of image
        # a_patch_corners = [item for item in a_patch_corners
        # if cls.is_legal_patch(item, image_size)]

        # get four rotated patches
        half_csize = int(cls.CORE_SIZE/2)
        b_patch_centers = (
            np.array(
                [[core_pos[0] + half_csize, core_pos[1]],
                    [core_pos[0] + cls.CORE_SIZE, core_pos[1] + half_csize],
                    [core_pos[0] + half_csize, core_pos[1] + cls.CORE_SIZE],
                    [core_pos[0], core_pos[1] + half_csize]]))

        angles = [45., 135., 225., 315.]

        for i in range(4):
            M = cv2.getRotationMatrix2D(
                tuple(b_patch_centers[i]), angles[i], 1.0)
            rotated = cv2.warpAffine(
                image, M, image_size, borderMode=cv2.INTER_CUBIC)
            patch = cv2.getRectSubPix(
                rotated, tuple(patch_size), tuple(b_patch_centers[i]))
            patches.append(patch)

        return patches

    @classmethod
    def get_cores_from_proposals(cls, prop_pos_list, prop_size):
        """
        this function returns some cores from a group of proposals,
        i.e. sliding windows
        that are classified to be positive. Should cores have overlapping?
        """

        print("get_cores_from_proposals: let's make it simple at first!")
        print("Assumption:\n", "1. Proposal(square) size is integer multiple of core(square) size\n",
              "2. Proposal is sliding by distances that are integer multiple of core(square) size")

        core_pos_list = list()

        for prop_pos in prop_pos_list:
            x_list = np.arange(
                prop_pos[0], prop_pos[0] + prop_size, cls.CORE_SIZE).tolist()
            y_list = np.arange(
                prop_pos[1], prop_pos[1] + prop_size, cls.CORE_SIZE).tolist()

            two_lists = [x_list, y_list]

            core_pos_list.extend(list(itertools.product(*two_lists)))

        core_pos_set = set(core_pos_list)

        return core_pos_set

    @staticmethod
    def is_legal_patch(patch_corners, image_size):
        for corner in patch_corners:
            if corner[0] < 0 or corner[1] < 0 or corner[0] >= image_size[0] or corner[1] >= image_size[1]:
                return False
        return True

    @staticmethod
    def inference_over_patches(patch_scores):
        """t
        his function would do a simple inference
        based on scores of patches around the
        core region of interest. Currently this 'inference' is just average
        """

        core_score = np.average(patch_scores)
        # expecting patch_score to be a numpy array

        return core_score

    @classmethod
    def plot_core_scores(cls, color_img, core_pos_and_score):

        output = color_img.copy()
        for pos, score in core_pos_and_score:
            overlay = color_img.copy()
            alpha = np.sqrt(score - 0.5)
            cv2.rectangle(overlay, pos,
                          (pos[0] + cls.CORE_SIZE, pos[1] + cls.CORE_SIZE),
                          (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        return output
