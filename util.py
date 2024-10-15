import pickle

from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("./model/model.p", "rb"))


def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY
# 1. empty or not (spot_bgr):
# This function takes an image of a parking area and predicts 
# whether this area is empty or occupied. 
# It uses a ready-made model for this prediction.

# spot_bgr : image of parking lot
# img_resized : we make the parking lot smaller. because I want our model to work faster.
# img_resized_flatten() : we converted the resized image into a single vector. we made it suitable for the input format of the model.
# if the estimate is 0, the parking lot is empty. if it is 1, the parking lot is full.
 
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

