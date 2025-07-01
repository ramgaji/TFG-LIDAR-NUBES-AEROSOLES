"""
nubestxt.py

Procesa todos los .nc de `data/`:  
1. Carga `cloud_base_height` (288 tiempos x hasta 3 capas),  a 3 columnas.  
2. Divide los 288 instantes en 64 bloques y calcula la media de cada bloque por capa.  
3. Extrae con autoencoder + KMeans la máscara de nube y, para cada bloque,  
   determina la **altura de base de nube** como la capa más baja (mayor índice del array de nube)  
   donde el cluster asignado es 'nube'.  

Guarda en `salidastxt/nubes_multilayer.txt` las columnas:
"""

import os
import numpy as np
import xarray as xr
from skimage.transform import resize
from tensorflow.keras.models import load_model
import joblib
import datetime

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
data_path    = '../data'
encoder_path = 'dec/encoder.keras'
kmeans_path  = 'dec/kmeans.pkl'
station_alt  = 680.0       # altitud de la estación en metros

# resolución de la rejilla reducida y número de timesteps
n_layers, n_times = 32, 64
# duración aproximada de cada bloque en minutos
dt = datetime.timedelta(minutes=24*60/n_times)

# carpeta donde guardaremos el nuevo TXT
txt_path = 'salidastxt'
os.makedirs(txt_path, exist_ok=True)

# función para extraer la fecha del nombre L2_..._AYYYYMMDD.nc
def extract_date(fname: str) -> datetime.datetime:
    part = fname.split('_A')[1].split('.')[0]
    return datetime.datetime.strptime(part, '%Y%m%d')

# cargamos el encoder y el kmeans una sola vez
encoder = load_model(encoder_path)
cluster = joblib.load(kmeans_path)

# constantes para el mapeo de clusters a categorías (nubes=0, resto=1)
category_mapping = {
    1: 7,  3: 7, 5: 1, 4: 5,
    2: 0,  6: 0, 0: 1, 7: 1
}
CLOUD_CAT = 0

# acumuladores
all_times    = []
all_layer1   = []
all_layer2   = []
all_layer3   = []
all_cbase    = []

# recorremos todos los archivos .nc
for nc_file in sorted(os.listdir(data_path)):
    if not nc_file.endswith('.nc'):
        continue

    date0 = extract_date(nc_file)
    print(f"Procesando {nc_file}")

    # --- 1) leemos cloud_base_height y lo reducimos a 3 capas si hace falta
    ds = xr.open_dataset(os.path.join(data_path, nc_file))
    cbh = ds['cloud_base_height'].values  # forma (T,) o (T,3)
    ds.close()
    # aseguramos forma (T, 3)
    if cbh.ndim == 1:
        cbh = cbh.reshape(-1, 1)
    # si tiene menos de 3 capas, rellenamos con NaN
    if cbh.shape[1] < 3:
        pad = np.full((cbh.shape[0], 3 - cbh.shape[1]), np.nan)
        cbh = np.hstack([cbh, pad])

    # dividimos el índice temporal en n_times bloques
    idxs = np.arange(cbh.shape[0])
    blocks = np.array_split(idxs, n_times)

    # para cada capa j=0,1,2, calculamos la media por bloque si mayoría no-nan
    cbh_red = np.full((n_times, 3), np.nan)
    for j in range(3):
        for i, blk in enumerate(blocks):
            vals = cbh[blk, j]
            count_valid = np.sum(~np.isnan(vals))
            if count_valid > len(blk) / 2:
                cbh_red[i, j] = np.nanmean(vals)

    # --- 2) leemos altitud y la reducimos a n_layers
    ds = xr.open_dataset(os.path.join(data_path, nc_file))
    alt = ds['altitude'].values
    ds.close()
    alt = np.flip(np.maximum(alt - station_alt, 0.0), axis=0)
    alt_grid = resize(alt, (n_layers,), order=1,
                      preserve_range=True, anti_aliasing=False)

    # --- 3) preprocess backscatter y extraer clusters
    ds = xr.open_dataset(os.path.join(data_path, nc_file))
    rcs = ds['attenuated_backscatter_0'].load().values
    ds.close()
    data = np.log(rcs)
    data = np.flip(data, axis=0)
    data = np.clip(data, -2, 2)
    np.nan_to_num(data, nan=-2, copy=False)
    img = 1 - ((data + 2) / 4.0)

    img_rs  = resize(img, (256, 512), order=0, anti_aliasing=False)
    xenc    = img_rs[np.newaxis, ..., np.newaxis]
    encoded = encoder.predict(xenc)[0]
    flat    = encoded.reshape(-1, encoded.shape[-1])
    lbl_img = cluster.predict(flat).reshape(encoded.shape[:2])

    # reducimos a la rejilla (n_layers x n_times) y mapeamos categorías
    small_lbl = resize(lbl_img, (n_layers, n_times),
                       order=0, preserve_range=True, anti_aliasing=False).astype(int)
    mapped = np.vectorize(lambda x: category_mapping.get(x, -1))(small_lbl)

    # --- 4) extraemos para cada timestep: time_index, layer1,2,3, cluster_base
    for i in range(n_times):
        tstr = (date0 + i*dt).strftime('%Y-%m-%d %H:%M')
        all_times.append(tstr)
        all_layer1.append(cbh_red[i, 0])
        all_layer2.append(cbh_red[i, 1])
        all_layer3.append(cbh_red[i, 2])

        # rows donde se detecta nube
        rows = np.where(mapped[:, i] == CLOUD_CAT)[0]
        if rows.size > 0:
            # base de nube = capa más baja = índice máximo
            crow = int(rows.max())
            cbase = float(alt_grid[crow])
        else:
            cbase = np.nan
        all_cbase.append(cbase)

# --- 5) escribimos el TXT multilayer
out_txt = os.path.join(txt_path, 'nubes_multilayer.txt')
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write('time_index layer1 layer2 layer3 cbase\n')
    for t, l1, l2, l3, h in zip(all_times, all_layer1, all_layer2, all_layer3, all_cbase):
        l1s = f'{l1:.2f}' if not np.isnan(l1) else 'nan'
        l2s = f'{l2:.2f}' if not np.isnan(l2) else 'nan'
        l3s = f'{l3:.2f}' if not np.isnan(l3) else 'nan'
        hs  = f'{h:.2f}'  if not np.isnan(h)  else 'nan'
        f.write(f'{t} {l1s} {l2s} {l3s} {hs}\n')

print(f' Multilayer year series guardada en {out_txt}')





