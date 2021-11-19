import imutils
import cv2
#from google.colab.patches import cv2_imshow

image = cv2.imread('D:/SOLECTHON/keypointextraction/blue.jpg')
# cv2.imshow('',image)


lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(lab)
cv2.imshow('',lab) 

cv2.imshow('',L)

      
ret, thresh1 = cv2.threshold(A, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)      
   
cv2.imshow('',thresh1)

#print(ret, thresh1)

# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
# print(c)
# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])
rng = int((extBot[1] - extTop[1]) * 0.2)

finalYBot = extBot[1] + rng
finalYTop = extBot[1] - rng

newC = []
for i in c:
  if i[0][1] >= finalYTop and i[0][1] <= finalYBot  : 
    newC.append((i[0][0],i[0][1]))
 

extLeft = min(newC)
extRight = max(newC)

cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 5, (0, 0, 255), -1)
cv2.circle(image, extRight, 5, (0, 255, 0), -1)
cv2.circle(image, extTop, 5, (0, 0, 255), -1)
cv2.circle(image, extBot, 8, (255, 255, 0), -1)


centerBase = ((extLeft[0]+extRight[0])//2 ,(extLeft[1]+extRight[1])//2 )
cv2.circle(image, centerBase, 1, (255, 255, 255), -1)


print("top", extTop )
print("base center",centerBase)
print("left",extLeft)
print("right",extRight)
# show the output image
cv2.imshow('',image)



import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imread 
from mpl_toolkits.mplot3d import Axes3D 
import scipy.ndimage as ndimage 
 
imageFile = 'blue.jpg' 
mat = imread(imageFile) 
mat = mat[:,:,0] # get the first channel 
rows, cols = mat.shape 
xv, yv = np.meshgrid(range(cols), range(rows)[::-1]) 
 
blurred = ndimage.gaussian_filter(mat, sigma=(5, 5), order=0) 
fig = plt.figure(figsize=(6,6)) 
 
ax = fig.add_subplot(221) 
ax.imshow(mat, cmap='gray') 
 
ax = fig.add_subplot(222, projection='3d') 
ax.elev= 75 
ax.plot_surface(xv, yv, mat) 
 
ax = fig.add_subplot(223) 
ax.imshow(blurred, cmap='gray') 
 
ax = fig.add_subplot(224, projection='3d') 
ax.elev= 75 
ax.plot_surface(xv, yv, blurred) 
plt.show()

