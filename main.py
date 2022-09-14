import math

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

rgbImage = io.imread('shed1-small.jpg')
(m, n, o) = rgbImage.shape
# Extract color channels.
redChannel = rgbImage[:, :, 0]  # Red channel
print(redChannel)
greenChannel = rgbImage[:, :, 1]  # Green channel
blueChannel = rgbImage[:, :, 2]  # Blue channel
# Create an all black channel.
allBlack = np.zeros((m, n), dtype=np.uint8)
# Create color versions of the individual color channels.
justRed = np.stack((redChannel, allBlack, allBlack), axis=2)
justGreen = np.stack((allBlack, greenChannel, allBlack), axis=2)
justBlue = np.stack((allBlack, allBlack, blueChannel), axis=2)
# Recombine the individual color channels to create the original RGB image again.
recombinedRGBImage = np.stack((redChannel, greenChannel, blueChannel), axis=2)
plt.imshow(recombinedRGBImage)
plt.show()
print(justRed.shape)
print(justRed.dtype)
print(justRed.size)
print(justRed[0:5, 0:5, 0])
io.imsave('justRed1.jpg', justRed)
#io.imsave("RedChannel.jpg", redChannel)
io.imshow(justRed)
plt.title('Red Channel')
plt.show()

io.imsave("justGreen1.jpg", justGreen)
io.imshow(justGreen)
plt.title('Green Channel')
plt.show()

io.imsave("justBlue1.jpg", justBlue)
io.imshow(justBlue)
plt.title('blue Channel')
plt.show()

AG = (redChannel/3 + greenChannel/3 + blueChannel/3)
io.imsave("AGChannel.jpg", AG)
io.imshow(AG, cmap = plt.get_cmap("gray"))
plt.title("AG")
plt.show()
print(AG[0:10,0:10])

# image values not normalized to be in 0.0 to 1.0) range
#imageFloat1 = rgbImage.astype(np.float32)
#plt.imshow(imageFloat1)
#plt.show()
#print(imageFloat1[0:5, 0:5, 0])

# normalize the image and display
#from skimage import img_as_float

#floatImage2 = img_as_float(rgbImage)
#plt.imshow(floatImage2)
#plt.show()
#print(floatImage2[0:5, 0:5, 0])

#for i in range(5, 10):
#    for j in range(20, 25):
#        print(floatImage2[i, j, 0])
#        print('\n')

# compute the histogram
[x, y] = redChannel.shape
redhist = np.zeros(256, dtype=np.intc)
for i in range(x):
    for j in range(y):
        redhist[rgbImage[i, j, 0]] += 1
plt.plot(redhist)
plt.title("Red Channel Histogram")
plt.show()

[x, y] = greenChannel.shape
greenhist = np.zeros(256, dtype=np.intc)
for i in range(x):
    for j in range(y):
        greenhist[rgbImage[i, j, 0]] += 1
plt.plot(greenhist)
plt.title("Green Channel Histogram")
plt.show()

[x, y] = blueChannel.shape
bluehist = np.zeros(256, dtype=np.intc)
for i in range(x):
    for j in range(y):
        bluehist[rgbImage[i, j, 0]] += 1
plt.plot(bluehist)
plt.title("Blue Channel Histogram")
plt.show()

hist = np.zeros(256, dtype=np.intc)
for i in range(m):
    for j in range(n):
        hist[rgbImage[i, j, 0]] += 1

val_5 = int(input("Enter your threshold value for Binarizing the image: "))
print(val_5)
AB = np.zeros((m,n), dtype= np.uint8)
for i in range(m):
    for j in range(n):
        if AG[i,j] >= val_5:
            AB[i,j] = 255
        else:
            AB[i,j] = 0

io.imshow(AB, cmap = plt.get_cmap("gray"))
plt.title("AB")
plt.show()

GM = np.zeros((m,n), dtype= np.uint8)
Gx=0
Gy=0
for i in range(m):
    for j in range(n):
        if i!=m-1:
            Gy = AG[i+1,j] - AG[i,j]
        else:
            Gy = 0
        if j != n-1:
            Gx = AG[i, j+1] - AG[i, j]
        else:
            Gx = 0
        GM[i,j] = math.sqrt(Gx*Gx+Gy*Gy)

val_6 = int(input("Enter your threshold value for Simple edge detection: "))
print(val_6)
AE = np.zeros((m, n), dtype=np.uint8)
for i in range(m):
    for j in range(n):
        if GM[i,j]>val_6:
            AE[i,j]=255
        else:
            AE[i,j]=0
io.imshow(AE, cmap=plt.get_cmap("gray"))
plt.title("AE")
plt.show()

k = int(m/2)
l = int(n/2)
AG2 =  np.zeros((k,l))
for i in range(k):
    for j in range(l):
        AG2[i,j]=(AG[i*2,j*2]+AG[(i*2)+1,j*2]+AG[i*2,(j*2)+1]+AG[(i*2)+1,(j*2)+1])/4
io.imshow(AG2, cmap=plt.get_cmap("gray"))
plt.title("AG2")
plt.show()


k = int(m/4)
l = int(n/4)
AG4 =  np.zeros((k,l))
for i in range(k):
    for j in range(l):
        AG4[i,j]=(AG2[i*2,j*2]+AG2[(i*2)+1,j*2]+AG2[i*2,(j*2)+1]+AG2[(i*2)+1,(j*2)+1])/4
io.imshow(AG4, cmap=plt.get_cmap("gray"))
plt.title("AG4")
plt.show()

k = int(m/8)
l = int(n/8)
AG8 =  np.zeros((k,l))
for i in range(k):
    for j in range(l):
        AG8[i,j]=(AG4[i*2,j*2]+AG4[(i*2)+1,j*2]+AG4[i*2,(j*2)+1]+AG4[(i*2)+1,(j*2)+1])/4
io.imshow(AG8, cmap=plt.get_cmap("gray"))
plt.title("AG8")
plt.show()

































#print(hist)

#import matplotlib.pyplot as plt

# plot example
#x = np.arange(0, 5, 0.1)
#y = np.sin(x)
#plt.plot(x, y)
#plt.show()

# plot the histogram
#plt.plot(hist)
#plt.show()

#plt.imshow(greenChannel, cmap=plt.cm.gray)
#plt.show()

#plt.imshow(redChannel, cmap=plt.cm.gray)
#plt.show()

#import matplotlib.pyplot as plt

#from skimage import data
#from skimage.color import rgb2gray

#original = data.astronaut()
#grayscale = rgb2gray(original)

#fig, axes = plt.subplots(1, 2, figsize=(8, 4))
#ax = axes.ravel()

#ax[0].set_title("Original")
#ax[1].imshow(grayscale, cmap=plt.cm.gray)
#ax[1].set_title("Grayscale")

#fig.tight_layout()
#plt.show()
