import cv2
import numpy as np

Icolor = cv2.imread('images/mustafa.jpeg')

height = np.size(Icolor, 0)
width = np.size(Icolor, 1)

# Horizontal flip
HFlip = Icolor[:,::-1,:]
cv2.imwrite("HFlip.png", HFlip)

# Vertical flip
VFlip = Icolor[::-1,:,:]
cv2.imwrite("VFlip.png", VFlip)

# Split in 2
S2_left     = Icolor[:,:int(width/2),:]
S2_right    = Icolor[:,int(width/2):width,:]
S2          = np.concatenate((S2_right, S2_left), axis=1)
cv2.imwrite("S2.png", S2)

# Split in 4
S4_topleft      = Icolor[:int(height/2),:int(width/2):,:]
S4_topright     = Icolor[:int(height/2),int(width/2):width,:]
S4_bottomleft   = Icolor[int(height/2):height,:int(width/2):,:]
S4_bottomright  = Icolor[int(height/2):height,int(width/2):width,:]

S4_top      = np.concatenate((S4_bottomright, S4_bottomleft), axis=1)
S4_bottom   = np.concatenate((S4_topright, S4_topleft), axis=1)
S4          = np.concatenate((S4_top, S4_bottom), axis=0)
cv2.imwrite("S4.png", S4)