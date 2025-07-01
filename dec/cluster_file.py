import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model
import joblib
from skimage.transform import resize
import xarray as xr

# Input
data_path = '../data'
yyyy = '2024'
mm = '12'
dd = '16'
#station = '0-100-20000-0000001-A'
#station = '0-20000-0-07014-A'
#station = '0-20000-0-07110-A'
station = '0-20000-0-01492-A'

# Simulate partial measurements
t_max = 9
fill = 'cropping'  # 'cropping', 'padding'

method = 'kmeans'  # 'kmeans', 'hdbscan'
savefig = True

# Paths to saved models
encoder_path = 'dec/encoder.keras'
kmeans_path = 'dec/kmeans.pkl'

# Load the encoder and kmeans model
encoder = load_model(encoder_path)
if method == 'kmeans':
    cluster = joblib.load(kmeans_path)

# Parameters
var = 'attenuated_backscatter_0'
vmin, vmax = -2, 2  # Use the same limits as for plotting
target_size = (256, 512)

# Load and preprocess the data
da = xr.open_dataset(f'{data_path}/{yyyy}/{mm}/{dd}/AP_{station}-{yyyy}-{mm}-{dd}.nc')[var].load()
if t_max:
    if fill == 'padding':
        da = da.where(da.time.dt.hour <= t_max, np.nan)
    elif fill == 'cropping':
        da = da.where(da.time.dt.hour <= t_max, drop=True)

# Log transform
data = np.log(da)

# Transpose and flip
data = np.flip(data, axis=0)

# Clip data to the range [vmin, vmax] and then scale it
data_clipped = np.clip(data, vmin, vmax)

# Replace NaNs with minimum values
np.nan_to_num(data_clipped, copy=False, nan=vmin)

# Invert the normalized data to reflect the 'gray_r' colormap
data_normalized = (data_clipped - vmin) / (vmax - vmin)
data_inverted = 1 - data_normalized  # Invert the grayscale values

# Resize the inverted data to the target size
resized_data = resize(
    data_inverted,
    output_shape=target_size,
    order=0,  # Nearest-neighbor interpolation to avoid smoothing
    anti_aliasing=False
)

# Add batch and channel dimensions for further processing
data_array = np.expand_dims(np.expand_dims(resized_data, axis=0), axis=-1)

# Encode the image to get feature representation
encoded_img = encoder.predict(data_array)[0]  # Remove batch dimension
print(f"Encoded image shape: {encoded_img.shape}")

# Step 1: Aggregate Encoded Features
print('encoded_img', np.shape(encoded_img))
aggregated_encoded_img = np.mean(encoded_img, axis=-1)  # Aggregated to single-channel (16, 32)
print('aggregated_encoded_img', np.shape(aggregated_encoded_img))

# Optional: Normalize for better visualization
aggregated_encoded_img = (aggregated_encoded_img - aggregated_encoded_img.min()) / (aggregated_encoded_img.max() - aggregated_encoded_img.min())

# Step 2: Flatten Encoded Features and Cluster
encoded_img_flat = encoded_img.reshape(-1, encoded_img.shape[-1])  # Flatten spatial dimensions for clustering
pixel_labels = cluster.predict(encoded_img_flat)  # Get cluster labels for each pixel

# Step 3: Map clusters to categories
category_mapping = {
    1: 1,  # molecules
    3: 1,  # molecules
    5: 1,  # noise
    4: 1,  # aerosols
    2: 0,  # clouds
    6: 0,  # clouds
    0: 1,  # other
    7: 1   # other
}
pixel_labels = np.vectorize(category_mapping.get)(pixel_labels)

# Reshape the cluster labels back to the spatial dimensions
pixel_labels_image_shape = pixel_labels.reshape(encoded_img.shape[0], encoded_img.shape[1])

# Step 4: Upsample the cluster labels to match the original image size
upsampled_pixel_labels = resize(
    pixel_labels_image_shape,
    (target_size[0], target_size[1]),
    order=0,  # Nearest-neighbor interpolation
    preserve_range=True,
    anti_aliasing=False
)
upsampled_pixel_labels = upsampled_pixel_labels.astype(int)  # Ensure the labels are integers

# Step 5: Create a colormap for cluster labels
unique_labels = np.unique(upsampled_pixel_labels)
colormap = plt.get_cmap('tab20', len(unique_labels))  # Get a colormap with enough colors

# Step 6: Create an overlay image with transparency
clustered_image = colormap(upsampled_pixel_labels)  # Apply the colormap
clustered_image[..., 3] = 0.5  # Set the alpha channel to 0.5 for transparency

# Step 7: Plot the Results in a single figure with tighter layout
plt.figure(figsize=(16, 9))  # Adjust size as needed

# Set up a 2x2 grid with space for an additional 3x3 grid (for the 9 first features)
outer_grid = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1], wspace=0.1, hspace=0.1)

# Original image in top-left corner
ax1 = plt.subplot(outer_grid[0, 0])
ax1.imshow(data_array[0].reshape(target_size[0], target_size[1]), cmap='gray')
ax1.set_title("Original Image", fontsize=10)
ax1.axis('on')

# Overlay of clustered image in top-right corner
ax2 = plt.subplot(outer_grid[0, 1])
ax2.imshow(data_array[0].reshape(target_size[0], target_size[1]), cmap='gray')
ax2.imshow(clustered_image[:, :, :3], alpha=0.5)  # Overlay with transparency
ax2.set_title("Overlay of Clustered Image", fontsize=10)
ax2.axis('on')

# Upsampled Clustered Image in bottom-right corner
ax4 = plt.subplot(outer_grid[1, 1])
ax4.imshow(upsampled_pixel_labels, cmap='tab20')  # Adjust colors to distinguish clusters
ax4.set_title("Upsampled Clustered Image", fontsize=10)
ax4.axis('on')

# Map cluster labels to colors for the legend
cluster_colors = {label: colormap(idx) for idx, label in enumerate(unique_labels)}

"""# Create a legend with cluster colors
legend_elements = [
    plt.Line2D([0], [0], marker='o', color=colormap(idx)[:3], label=f'Cluster {label}',
               markersize=8, linestyle='None') for idx, label in enumerate(unique_labels)
]

# Add the legend below the clustered image
ax4.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
"""
# 3x3 grid of first 9 encoded features in the bottom-left corner
inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer_grid[1, 0], wspace=0.05, hspace=0.05)
for i in range(9):
    ax = plt.Subplot(plt.gcf(), inner_grid[i])
    ax.imshow(encoded_img[..., i], cmap='gray')
    ax.set_title(f'Feature {i+1}/{np.shape(encoded_img)[2]}', fontsize=8)
    ax.axis('off')
    plt.gcf().add_subplot(ax)

plt.tight_layout(pad=0.5)  # Further reduce padding around all panels
if savefig:
    plt.savefig(f'images/output/{yyyy}{mm}{dd}-{station}.png', bbox_inches='tight')
else:
    plt.show()