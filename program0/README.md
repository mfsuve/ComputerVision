Program 0
=============================

For all of the questions, the code needs to read this original picture named as mustafa.jpeg in order to get the following result images. The codes for the first, the second and the third questions are Q1.py, Q2.py and Q3.py, respectively.

Original Picture

1. I first read the picture of myself using cv2.imread function. For each channel, I
copied the original array of that picture and modified it. The array of my picture has 3
dimension. The first 2 is the size of that picture and the last dimension has a size of 3
which stores the RGB values. When I want to display channels, I clear the other ones.

Red Green Blue



For the images, 0 represents blue color, 1 represents green color and 2 represents red color at the 3rd dimensions of each array. For example I need to clear all the zeroth and the first elements of the array in order to get the red channel. The similar approach is used for the combining channels. ''Red+Green'' ''Red+Blue'' ''Green+Blue''

2. For getting an average grayscale image, I got the average of the 3 elements (RGB) in
the arrays at the 3rd dimension and filled a new 2 dimensional array with that values.
 Grayscale Image
The max value in that image is 249.0 and the min value is 0.66. In order to find how
many bits we need to use to represent this image, I used this formula:

log<sub>2</sub>(𝑚𝑎𝑥 − 𝑚𝑖𝑛) = 7.956

Hence, the minimum possible number of bits to represent this image (an integer) is 8.
3. In Q3.py file, I manipulated the array of the original picture to get these results:
Horizontal Flip Vertical Flip Slice in 2
Slice in 4

