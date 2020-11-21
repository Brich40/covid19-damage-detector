import cv2 as cv
import numpy as np
from bin.classes.image_process import ImageProcess

IMAGE_PATH = '/home/obr01/Documents/CERI/M2_S3/Traitement-Document/Projet/data/images/covid19.jpg'


def show_image(path):
    image = cv.imread(path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # show_image(IMAGE_PATH)
    ImageProcess.image_edge(IMAGE_PATH)
