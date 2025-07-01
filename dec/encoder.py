import os
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(3)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D

# Add seed for reproductability
tf.random.set_seed(42)

# Path to the images
training_image_dir = 'images/training'
validation_image_dir = 'images/validation'
checkpoints_dir = 'dec/checkpoints.weights.h5'
image_size = (256, 512)  # Resize images to a consistent size

# Step 1: Load and Preprocess the Dataset
training_images = []
for filename in os.listdir(training_image_dir):
    img_path = os.path.join(training_image_dir, filename)
    img = load_img(img_path, target_size=image_size, color_mode='grayscale')  # Load as grayscale
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    training_images.append(img_array)

validation_images = []
for filename in os.listdir(validation_image_dir):
    img_path = os.path.join(validation_image_dir, filename)
    img = load_img(img_path, target_size=image_size, color_mode='grayscale')  # Load as grayscale
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    validation_images.append(img_array)

training_images = np.array(training_images)
validation_images = np.array(validation_images)

print(f"Loaded dataset shape: {training_images.shape}")

# Step 2: Define and Train an Autoencoder
# Encoder
input_img = Input(shape=(256, 512, 1))  # 256, 512, 1
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) # 256, 512, 32
x = MaxPooling2D((2, 2), padding='same')(x) # 128, 256, 32
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # 128, 256, 64
x = MaxPooling2D((2, 2), padding='same')(x) # 64, 128, 64

# Latent Space
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x) # 64, 128, 128
encoded = MaxPooling2D((2, 2), padding='same')(x) # 32, 64, 128

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) # 32, 64, 128
x = UpSampling2D((2, 2))(x) # 64, 128, 128
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # 64, 128, 64
x = UpSampling2D((2, 2))(x) # 128, 256, 64
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) # 128, 256, 32
x = UpSampling2D((2, 2))(x) # 256, 512, 32
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # 256, 512, 1

# Model definition
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoints_dir, save_weights_only = True, monitor = 'val_loss', mode = 'min', verbose = 2, save_best_only = True)
autoencoder.fit(training_images, training_images, validation_data=(validation_images, validation_images), epochs=10, batch_size=16, shuffle=True, callbacks = [checkpoint])

# 1. Define the encoder model
encoder = Model(inputs=autoencoder.input, outputs=encoded)
encoder.save('dec/encoder.dev.keras')
