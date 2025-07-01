# imports 
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

img_path = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-02.png'
img_path = 'images/landscape.jpeg'

plt.rcParams["figure.figsize"] = (12, 50) 

# Path to the images
image_dir = 'images/rcs'
image_size = (512, 512)  # Resize images to a consistent size

# load and resize image to reduce processing time
img = cv.imread('images/rcs/AP_0-100-20000-0000001-A-2024-07-02.png')
img_resized = cv.resize(img, image_size, interpolation=cv.INTER_AREA) 

# Flatten the resized image to a list of RGB values
Z = img_resized.reshape((-1, 3))
Z = np.float32(Z)  # Convert to float32 for DBSCAN processing

# Set DBSCAN parameters
eps = 10  # Adjust as needed
min_samples = 50  # Adjust as needed
db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)

# Use labels to color the image
labels = db.labels_
unique_labels = np.unique(labels)

# Create a color map
# If there are only noise points (-1), set a single black color
if len(unique_labels) == 1 and unique_labels[0] == -1:
    colors = np.array([[0, 0, 0]])  # Only noise, all black
else:
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))  # Colors for clusters and noise
    colors[unique_labels == -1] = [0, 0, 0]  # Set noise (label -1) to black

# Map each label to its corresponding color
label_color_map = np.array([colors[unique_labels == label][0] for label in labels])

# Reshape the color-mapped labels to the resized image shape
clustered_img = label_color_map.reshape(img_resized.shape)

# Display the clustered result
cv.imshow('DBSCAN Clustered Image', clustered_img.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()