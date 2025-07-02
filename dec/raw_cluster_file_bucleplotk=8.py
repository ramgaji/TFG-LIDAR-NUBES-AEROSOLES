# -*- coding: utf-8 -*-
"""
raw_cluster_file_bucleplotk=8.py

Este script recorre todos los archivos LIDAR (.nc) en `data/`, aplica preprocesamiento,
usa un autoencoder + KMeans (k=8) para segmentar la imagen en 8 clusters,
y genera plots combinando la imagen original, la máscara de clusters y algunas features internas.
Los resultados se guardan en la carpeta `images/`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import joblib
from skimage.transform import resize
import xarray as xr

#   ruta de datos:
data_path = './data/'

#  Obtener la lista de archivos en la carpeta del año seleccionado
files = sorted([f for f in os.listdir(f"{data_path}") if f.endswith(".nc")])

#  Bucle para procesar cada archivo .nc
for nc_file in files:
    print(f" Procesando archivo: {nc_file}")

    #  Cargar los datos desde el archivo seleccionado
    var = 'attenuated_backscatter_0'  # Variable a analizar
    ds = xr.open_dataset(f"{data_path}/{nc_file}")[var].load()

    # Preprocesamiento
    vmin, vmax = -2, 2
    target_size = (256, 512)

    # Log transform y normalización
    data = np.log(ds)
    data = np.flip(data, axis=0)
    data_clipped = np.clip(data, vmin, vmax)
    np.nan_to_num(data_clipped, copy=False, nan=vmin)

    # Normalizar e invertir los valores para la visualización
    data_normalized = (data_clipped - vmin) / (vmax - vmin)
    data_inverted = 1 - data_normalized

    # Redimensionar los datos
    resized_data = resize(
        data_inverted,
        output_shape=target_size,
        order=0,
        anti_aliasing=False
    )

    # Preparar la entrada para el modelo
    data_array = np.expand_dims(np.expand_dims(resized_data, axis=0), axis=-1)

    # Cargar modelos preentrenados
    encoder_path = 'dec/encoder.keras'
    kmeans_path = 'dec/kmeans.pkl'

    encoder = load_model(encoder_path)
    cluster = joblib.load(kmeans_path)

    # Extraer características del modelo
    encoded_img = encoder.predict(data_array)[0]

    #  Clustering
    encoded_img_flat = encoded_img.reshape(-1, encoded_img.shape[-1])
    pixel_labels = cluster.predict(encoded_img_flat)
    pixel_labels_image_shape = pixel_labels.reshape(encoded_img.shape[0], encoded_img.shape[1])

    #  Upsample para igualar la imagen original
    upsampled_pixel_labels = resize(
        pixel_labels_image_shape, 
        (target_size[0], target_size[1]), 
        order=0, 
        preserve_range=True, 
        anti_aliasing=False
    ).astype(int)

    #  Crear colormap
    unique_labels = np.unique(upsampled_pixel_labels)
    colormap = plt.get_cmap('tab20', len(unique_labels))

    # Visualización
    plt.figure(figsize=(16, 9))
    outer_grid = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1], wspace=0.1, hspace=0.1)

    ax1 = plt.subplot(outer_grid[0, 0])
    ax1.imshow(data_array[0].reshape(target_size[0], target_size[1]), cmap='gray')
    ax1.set_title("Original Image", fontsize=10)
    ax1.axis('on')

    ax2 = plt.subplot(outer_grid[0, 1])
    ax2.imshow(data_array[0].reshape(target_size[0], target_size[1]), cmap='gray')
    ax2.imshow(colormap(upsampled_pixel_labels)[:, :, :3], alpha=0.5)
    ax2.set_title("Clustered Overlay", fontsize=10)
    ax2.axis('on')

    ax4 = plt.subplot(outer_grid[1, 1])
    ax4.imshow(upsampled_pixel_labels, cmap='tab20')
    ax4.set_title("Upsampled Clustered Image", fontsize=10)
    ax4.axis('on')
    
    #  Agregar las features (características) de la imagen codificada
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer_grid[1, 0], wspace=0.05, hspace=0.05)
    for i in range(9):  # Mostrar las primeras 9 características
        ax = plt.Subplot(plt.gcf(), inner_grid[i])
        ax.imshow(encoded_img[..., i], cmap='gray')
        ax.set_title(f'Feature {i+1}/128', fontsize=8)
        ax.axis('off')
        plt.gcf().add_subplot(ax)
        
    #  Guardar la imagen en su carpeta correspondiente
    output_folder = f"images"
    os.makedirs(output_folder, exist_ok=True)

    output_path = f"{output_folder}/{nc_file.replace('.nc', '')}_rawk=8.png"

    #  Eliminar si ya existe 
    if os.path.exists(output_path):
        os.remove(output_path)

    #  Guardar la imagen
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f" Imagen guardada en: {output_path}")

print(" Procesamiento completado para todos los archivos")
