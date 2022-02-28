import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image                                                                                

####create a matrix of random colors
filename = "Refer"

matrix=np.random.random((4,4,3))
nx,ny,nz=np.shape(matrix)
CXY=np.zeros([ny, nx])
for i in range(ny):
    for j in range(nx):
        CXY[i,j]=np.max(matrix[j,i,:])

#Save binary data
np.save(filename + '.npy', CXY)
print(filename + " was saved")

'''

#Load npy
img_array = np.load(filename + '.npy')
plt.imshow(img_array)


####Save npy as png
filename = "original-image"

img_name = filename +".png"
matplotlib.image.imsave(img_name, img_array)
print(filename + " was saved")


#### Convert that png back to numpy array

img = Image.open( filename + '.png' )
data = np.array( img, dtype='uint8' )

#Convert the new npy file to png
filename = "new-array"

np.save( filename + '.npy', data)
print(filename + " was saved")


#Load npy
img_array = np.load(filename + '.npy')

filename = "new-image"
#Save as png
img_name = filename +".png"
matplotlib.image.imsave(img_name, img_array)
print(filename + " was saved")
'''