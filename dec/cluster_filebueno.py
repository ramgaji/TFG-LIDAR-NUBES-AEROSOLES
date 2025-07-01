import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model
import joblib
from skimage.transform import resize
import xarray as xr

# ------------------------------
# CONFIGURACIÓN DE ENTRADA
# ------------------------------
data_path = '../data'
yyyy = '2024'
mm = '12'
dd = '16'
station = '0-20000-0-01492-A'

# Parámetros de preprocesado
t_max = None          # Igual que en raw, para no recortar
fill = 'cropping'     # Da igual si t_max=None
method = 'kmeans'
savefig = True

# Rutas a modelos
encoder_path = 'dec/encoder.keras'
kmeans_path = 'dec/kmeans.pkl'

# ------------------------------
# CARGA DE MODELOS
# ------------------------------
encoder = load_model(encoder_path)
cluster = joblib.load(kmeans_path)

# ------------------------------
# PARÁMETROS DE PROCESAMIENTO
# ------------------------------
var = 'attenuated_backscatter_0'
vmin, vmax = -2, 2
target_size = (256, 512)

# ------------------------------
# CARGA Y PREPROCESAMIENTO DE DATOS (igual que raw)
# ------------------------------
nc_file = f'AP_{station}-{yyyy}-{mm}-{dd}.nc'
ds = xr.open_dataset(f'{data_path}/{yyyy}/{mm}/{dd}/{nc_file}')
da = ds[var].load()

if t_max:
    if fill == 'padding':
        da = da.where(da.time.dt.hour <= t_max, np.nan)
    elif fill == 'cropping':
        da = da.where(da.time.dt.hour <= t_max, drop=True)

# Mismo pipeline que “raw”
data = np.log(da)
data = np.flip(data, axis=0)

data_clipped = np.clip(data, vmin, vmax)
np.nan_to_num(data_clipped, copy=False, nan=vmin)

data_normalized = (data_clipped - vmin) / (vmax - vmin)
data_inverted = 1 - data_normalized

resized_data = resize(
    data_inverted,
    output_shape=target_size,
    order=0,
    anti_aliasing=False
)

data_array = np.expand_dims(np.expand_dims(resized_data, axis=0), axis=-1)

# ------------------------------
# CODIFICACIÓN Y CLUSTERING
# ------------------------------
encoded_img = encoder.predict(data_array)[0]
encoded_img_flat = encoded_img.reshape(-1, encoded_img.shape[-1])
pixel_labels = cluster.predict(encoded_img_flat)

# (A) Mapeo a 0/1 si lo deseas
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

pixel_labels_image = pixel_labels.reshape(encoded_img.shape[0], encoded_img.shape[1])

upsampled_pixel_labels = resize(
    pixel_labels_image,
    (target_size[0], target_size[1]),
    order=0,
    preserve_range=True,
    anti_aliasing=False
).astype(int)

# ------------------------------
# CREACIÓN DE LA IMAGEN DE CLÚSTERES
# ------------------------------
unique_labels = np.unique(upsampled_pixel_labels)
colormap = plt.get_cmap('tab20', len(unique_labels))
clustered_image = colormap(upsampled_pixel_labels)
clustered_image[..., 3] = 0.5  # Transparencia

# ------------------------------
# PLOTEO DE RESULTADOS
# ------------------------------
plt.figure(figsize=(16, 9))
outer_grid = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1], wspace=0.1, hspace=0.1)

# Imagen original
ax1 = plt.subplot(outer_grid[0, 0])
ax1.imshow(data_array[0].reshape(target_size), cmap='gray')
ax1.set_title("Original Image", fontsize=10)
ax1.axis('on')

# Overlay
ax2 = plt.subplot(outer_grid[0, 1])
ax2.imshow(data_array[0].reshape(target_size), cmap='gray')
ax2.imshow(clustered_image[..., :3], alpha=0.5)
ax2.set_title("Overlay of Clustered Image", fontsize=10)
ax2.axis('on')

# Imagen de clústeres upsampled
ax4 = plt.subplot(outer_grid[1, 1])
ax4.imshow(upsampled_pixel_labels, cmap='tab20')
ax4.set_title("Upsampled Clustered Image", fontsize=10)
ax4.axis('on')

# Mostrar 9 features codificadas (opcional)
inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer_grid[1, 0], wspace=0.05, hspace=0.05)
for i in range(9):
    ax = plt.Subplot(plt.gcf(), inner_grid[i])
    ax.imshow(encoded_img[..., i], cmap='gray')
    ax.set_title(f'Feature {i+1}/{encoded_img.shape[2]}', fontsize=8)
    ax.axis('off')
    plt.gcf().add_subplot(ax)

plt.tight_layout(pad=0.5)
if savefig:
    out_file = f'images/output/{yyyy}{mm}{dd}-{station}_cluster_2cats.png'
    plt.savefig(out_file, bbox_inches='tight')
else:
    plt.show()
