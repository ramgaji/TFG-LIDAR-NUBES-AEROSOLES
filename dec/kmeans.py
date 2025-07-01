import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.autograph.set_verbosity(3)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans

# Path to the images
image_dir = 'images/k-training'
image_size = (256, 512)  # Resize images to a consistent size

# Step 1: Load and Preprocess the Dataset
images = []
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    img = load_img(img_path, target_size=image_size, color_mode='grayscale')  # Load as grayscale
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    images.append(img_array)

images = np.array(images)
print(f"Loaded dataset shape: {images.shape}")

# 1. Load encoder
encoder_path = 'dec/encoder.dev.keras'
encoder = load_model(encoder_path)

# 2. Encode each image to get pixel-level features
encoded_images = encoder.predict(images)

# Check shape of encoded images
print(f"Encoded images shape: {encoded_images.shape}")  # Expected (283, 16, 32, 128) for example

# 3. Flatten spatial dimensions but keep feature channels intact
# This will give (num_images * height * width, num_features)
num_images, enc_height, enc_width, num_features = encoded_images.shape
encoded_images_flat = encoded_images.reshape((num_images * enc_height * enc_width, num_features))
print(f"Encoded pixel features shape for clustering: {encoded_images_flat.shape}")

# 4. Apply KMeans to pixel features
kmeans = KMeans(n_clusters=8)  # Assuming 6 clusters for molecules, aerosols, clouds, etc.
pixel_labels = kmeans.fit_predict(encoded_images_flat)  # Clusters all pixels independently
joblib.dump(kmeans, 'dec/kmeans.dev.pkl')

# 5. Reshape pixel labels back into image form for visualization
# After clustering, reshape pixel_labels to (num_images, enc_height, enc_width)
pixel_labels_image_shape = pixel_labels.reshape((num_images, enc_height, enc_width))
print(f"Pixel-wise clustered image shape: {pixel_labels_image_shape.shape}")
