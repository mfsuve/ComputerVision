import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

color1 = cv2.imread('images/color1.png')
color2 = cv2.imread('images/color2.png')

# a)

## For color1
(r1Hist, bin) = np.histogram(np.ravel(color1[:, :, 2]), np.arange(257))
(g1Hist, bin) = np.histogram(np.ravel(color1[:, :, 1]), np.arange(257))
(b1Hist, bin) = np.histogram(np.ravel(color1[:, :, 0]), np.arange(257))
(r1PDF, bin) = np.histogram(np.ravel(color1[:, :, 2]), np.arange(257), density=True)
(g1PDF, bin) = np.histogram(np.ravel(color1[:, :, 1]), np.arange(257), density=True)
(b1PDF, bin) = np.histogram(np.ravel(color1[:, :, 0]), np.arange(257), density=True)
plt.figure(1)

plt.subplot(311)
plt.plot(r1Hist, 'r')
plt.title("histograms of the Color1 image")

plt.subplot(312)
plt.plot(g1Hist, 'g')

plt.subplot(313)
plt.plot(b1Hist, 'b')

plt.show()

## For color2
(r2Hist, bin) = np.histogram(np.ravel(color2[:, :, 2]), np.arange(257))
(g2Hist, bin) = np.histogram(np.ravel(color2[:, :, 1]), np.arange(257))
(b2Hist, bin) = np.histogram(np.ravel(color2[:, :, 0]), np.arange(257))
(r2PDF, bin) = np.histogram(np.ravel(color2[:, :, 2]), np.arange(257), density=True)
(g2PDF, bin) = np.histogram(np.ravel(color2[:, :, 1]), np.arange(257), density=True)
(b2PDF, bin) = np.histogram(np.ravel(color2[:, :, 0]), np.arange(257), density=True)
plt.figure(2)

plt.subplot(311)
plt.plot(r2Hist, 'r')
plt.title("histograms of the Color2 image")

plt.subplot(312)
plt.plot(g2Hist, 'g')

plt.subplot(313)
plt.plot(b2Hist, 'b')

plt.show()

def getCDF(arr):
    CDF = np.zeros((3, 256))  # 0: r, 1: g, 2: b
    for i in range(3):
        (PDF, bin) = np.histogram(np.ravel(arr[:, :, i]), np.arange(257), density=True)
        CDF[2 - i] = np.cumsum(PDF)
    return CDF

def histMatch(image, target):
    tCDF = getCDF(target)
    CDF  = getCDF(image)
    width  = np.size(image, axis=0)
    height = np.size(image, axis=1)

    LUT = np.zeros((3, 256))

    for channel in range(3):
        x = 0
        for i in range(0, 256):
            while x < 255 and tCDF[channel, x] < CDF[channel, i]:
                x += 1
            LUT[channel, i] = x

    for channel in range(3):
        for x in range(width):
            for y in range(height):
                image[x, y, 2 - channel] = LUT[channel, math.floor(image[x, y, 2 - channel])]

    return image

matched = histMatch(color1, color2)
plt.figure(3)
plt.title("histogram matched image")
plt.imshow(matched[:, :, ::-1])
plt.show()

(rHist, bin) = np.histogram(np.ravel(matched[:, :, 2]), np.arange(257))
(gHist, bin) = np.histogram(np.ravel(matched[:, :, 1]), np.arange(257))
(bHist, bin) = np.histogram(np.ravel(matched[:, :, 0]), np.arange(257))

plt.figure(4)

plt.subplot(311)
plt.plot(rHist, 'r')
plt.title("histograms of the matched image")

plt.subplot(312)
plt.plot(gHist, 'g')

plt.subplot(313)
plt.plot(bHist, 'b')

plt.show()











