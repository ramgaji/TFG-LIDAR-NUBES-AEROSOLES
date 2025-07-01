# imports 
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

img_path = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-02.png'
img_path = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-01.png'
#img_path = 'images/landscape.jpeg'

plt.rcParams["figure.figsize"] = (12, 50) 

# load image 
img = cv.imread(img_path)
print('img', np.shape(img))
Z = img.reshape((-1, 1))  # Flatten the image to a list of pixels
print('input img', np.shape(Z))

# Convert to np.float32
Z = np.float32(Z)

# Set the number of clusters (K)
K = 4
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(Z)  # Cluster each pixel
print('labels', np.shape(labels))

# Get the color centers for each cluster
centers = np.uint8(kmeans.cluster_centers_)
print('centers',np.shape(centers))

# Recolor each pixel based on its cluster label
res = centers[labels]
res2 = res.reshape(img.shape)  # Reshape to the original image dimensions

# Display the clustered result
cv.imshow('KMeans Clustered Image', res2)
cv.waitKey(0)
cv.destroyAllWindows()
