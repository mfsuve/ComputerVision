import cv2
import numpy as np
from matplotlib.pyplot import figure, title, imshow, show

def gradient(I):
    h = np.size(I, axis=0)
    w = np.size(I, axis=1)
    IxList = np.zeros((h, w))
    IyList = np.zeros((h, w))

    for x in range(1, h - 1):
        for y in range(1, w - 1):
            Ix = (int(I[x + 1, y]) - int(I[x - 1, y])) / 2
            Iy = (int(I[x, y + 1]) - int(I[x, y - 1])) / 2
            IxList[x, y] = Ix
            IyList[x, y] = Iy

    return IxList, IyList

def harris(Ix, Iy):
    t = 150
    size = 3
    halfsize = int(size / 2)
    h = np.size(Ix, axis=0)
    w = np.size(Ix, axis=1)
    newimage = np.zeros((h, w))

    for i in range(halfsize, h - halfsize):
        for j in range(halfsize, w - halfsize):
            Wx = Ix[i - halfsize: i + halfsize + 1, j - halfsize: j + halfsize + 1]
            Wy = Iy[i - halfsize: i + halfsize + 1, j - halfsize: j + halfsize + 1]
            G = np.zeros((2, 2))
            G[0, 0] = np.sum([x*x for x in Wx])
            G[0, 1] = np.sum([x*y for x in Wx for y in Wy])
            G[1, 0] = G[0, 1]   # no need to compute it again
            G[1, 1] = np.sum([y*y for y in Wy])
            if min(np.linalg.eig(G)[0]) > t:
                newimage[i, j] = 1
    return newimage

gray = cv2.imread('images/blocks.jpg', 0)
img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
figure(1)
title('Grayscale Image')
imshow(gray, cmap='gray')
show()

gray = cv2.GaussianBlur(gray, (5, 5), 1)    # blur image
figure(2)
title('Blurred image with 5x5 window, sigma=1')
imshow(gray, cmap='gray')
show()

Ix, Iy = gradient(gray)                     # take the gradients
corners = harris(Ix, Iy)                    # detect corners

corners = cv2.dilate(corners, None)

img[corners > 0] = [0, 255, 0]

figure(3)
title('Corners on top of the Image')
imshow(img)
show()
