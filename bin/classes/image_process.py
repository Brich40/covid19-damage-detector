"""
Name : image_process.py.py
Author : OBR01
contact : oussama.brich@gmail.com
Time    : 20/11/2020 23:45
Desc:
"""

import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt


class ImageProcess:

    @staticmethod
    def color_seg(choice, path):
        if choice == 'white':
            lower_hue = np.array([0, 0, 0])
            upper_hue = np.array([0, 0, 255])
        elif choice == 'black':
            lower_hue = np.array([0, 0, 0])
            upper_hue = np.array([130, 130, 130])
        return lower_hue, upper_hue

    @staticmethod
    def image_edge(path):
        # Take each frame
        frame = cv2.imread(path)
        # frame = cv2.imread('images/road_1.jpg')

        frame = imutils.resize(frame, height=300)
        chosen_color = 'black'

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of a color in HSV
        lower_hue, upper_hue = ImageProcess.color_seg(chosen_color, path)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_hue, upper_hue)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        edges = cv2.Canny(mask, 100, 200)

        # Remove black background
        tmp = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(edges)
        rgba = [b, g, r, alpha]
        edges_no_background = cv2.merge(rgba, 4)

        plt.subplot(122), plt.imshow(frame, cmap='gray'), plt.imshow(edges, cmap='gray')

        plt.show()

        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        cv2.waitKey(0)
