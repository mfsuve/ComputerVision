import cv2
from copy import deepcopy

Icolor = cv2.imread('images/mustafa.jpeg')

# Red Channel
red = deepcopy(Icolor)
red[:,:,0] = 0
red[:,:,1] = 0

cv2.imwrite("red.png", red)

# Green Channel
green = deepcopy(Icolor)
green[:,:,0] = 0
green[:,:,2] = 0

cv2.imwrite("green.png", green)

# Blue Channel
blue = deepcopy(Icolor)
blue[:,:,1] = 0
blue[:,:,2] = 0

cv2.imwrite("blue.png", blue)

# RedGreen Channel
redgreen = deepcopy(Icolor)
redgreen[:,:,0] = 0

cv2.imwrite("redgreen.png", redgreen)

# RedBlue Channel
redblue = deepcopy(Icolor)
redblue[:,:,1] = 0

cv2.imwrite("redblue.png", redblue)

# GreenBlue Channel
greenblue = deepcopy(Icolor)
greenblue[:,:,2] = 0

cv2.imwrite("greenblue.png", greenblue)