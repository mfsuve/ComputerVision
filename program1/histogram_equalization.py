import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

underexposed = cv2.imread('images/underexposed.jpg')

plt.figure(1)
plt.title('underexposed image')
plt.imshow(underexposed[:, :, ::-1])
plt.show()

r = underexposed[:, :, 2]
g = underexposed[:, :, 1]
b = underexposed[:, :, 0]

# a)
print("Red Channel mean:", np.mean(r), ", standard deviation:", np.std(r))
print("Green Channel mean:", np.mean(g), ", standard deviation:", np.std(g))
print("Blue Channel mean:", np.mean(b), ", standard deviation:", np.std(b))

# b)
r1 = np.ravel(r)
g1 = np.ravel(g)
b1 = np.ravel(b)

## Calculate Histograms
(rHist, bin) = np.histogram(r1, np.arange(257))
(gHist, bin) = np.histogram(g1, np.arange(257))
(bHist, bin) = np.histogram(b1, np.arange(257))

## Calculate PDF
(rPDF, bin) = np.histogram(r1, np.arange(257), density=True)
(gPDF, bin) = np.histogram(g1, np.arange(257), density=True)
(bPDF, bin) = np.histogram(b1, np.arange(257), density=True)

plt.figure(2)

plt.subplot(311)
plt.plot(rHist, 'r')
plt.axis([-10, 256, -500, 15000])
plt.title("histograms of the underexposed image")

plt.subplot(312)
plt.plot(gHist, 'g')
plt.axis([-10, 256, -500, 15000])

plt.subplot(313)
plt.plot(bHist, 'b')
plt.axis([-10, 256, -500, 15000])

plt.show()

# c)
rCDF = np.cumsum(rPDF)
gCDF = np.cumsum(gPDF)
bCDF = np.cumsum(bPDF)

plt.figure(3)

plt.subplot(311)
plt.plot(rCDF, 'r')
plt.axis([-10, 256, 0, 1])
plt.title("cdfs of the underexposed image")

plt.subplot(312)
plt.plot(gCDF, 'g')
plt.axis([-10, 256, 0, 1])

plt.subplot(313)
plt.plot(bCDF, 'b')
plt.axis([-10, 256, 0, 1])

plt.show()

# d)
# If the CDF is our LUT, then the histogram
# gets equalized. We need to multiply it by 255
# to preserve its intensity values.
def equalize(image):
    width = np.size(image, axis=0)
    height = np.size(image, axis=1)
    # get the PDF for red
    (rPDF, bin) = np.histogram(np.ravel(image[:, :, 2]), np.arange(257), density=True)
    # convert PDF to CDF and use it as LUT
    rLUT = 255 * np.cumsum(rPDF)
    for x in range(width):
        for y in range(height):
            image[x, y, 2] = rLUT[math.floor(image[x, y, 2])]

    # apply the same thing to the other channels
    (gPDF, bin) = np.histogram(np.ravel(image[:, :, 1]), np.arange(257), density=True)
    gLUT = 255 * np.cumsum(gPDF)
    for x in range(width):
        for y in range(height):
            image[x, y, 1] = gLUT[math.floor(image[x, y, 1])]

    (bPDF, bin) = np.histogram(np.ravel(image[:, :, 0]), np.arange(257), density=True)
    bLUT = 255 * np.cumsum(bPDF)
    for x in range(width):
        for y in range(height):
            image[x, y, 0] = bLUT[math.floor(image[x, y, 0])]

    return image

plt.figure(4)
plt.title('Brightness Equalized Image')
plt.imshow(equalize(underexposed[:, :, ::-1]))
plt.show()

## Calculate Histograms
(rHist, bin) = np.histogram(np.ravel(underexposed[:, :, 2]), np.arange(257))
(gHist, bin) = np.histogram(np.ravel(underexposed[:, :, 1]), np.arange(257))
(bHist, bin) = np.histogram(np.ravel(underexposed[:, :, 0]), np.arange(257))

plt.figure(5)

plt.subplot(311)
plt.plot(rHist, 'r')
plt.axis([-10, 256, 0, 15000])
plt.title("histograms of the equalized image")

plt.subplot(312)
plt.plot(gHist, 'g')
plt.axis([-10, 256, 0, 15000])

plt.subplot(313)
plt.plot(bHist, 'b')
plt.axis([-10, 256, 0, 15000])

plt.show()

# PDF is almost flattened, the image gets more illuminated
print("For illuminated image :")

# Red Channel mean: 130.593717228 , standard deviation: 69.6745152191
print("Red Channel mean:", np.mean(r), ", standard deviation:", np.std(r))
# Green Channel mean: 128.879831461 , standard deviation: 71.8700792653
print("Green Channel mean:", np.mean(g), ", standard deviation:", np.std(g))
# Blue Channel mean: 129.821797753 , standard deviation: 70.3049839773
print("Blue Channel mean:", np.mean(b), ", standard deviation:", np.std(b))

