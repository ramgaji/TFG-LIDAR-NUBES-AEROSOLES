"""
aerosoltxt.py

Para cada archivo .nc en `data/`:
 1. Carga la altitud y normaliza en una rejilla de 32 capas.
 2. Lee el backscatter atenuado, aplica log transform, clip y normalización.
 3. Usa un autoencoder y KMeans (k=8) para etiquetar cada píxel.
 4. Reduce el mapa de clusters a 32x64 y reasigna categorías (moléculas, aerosoles, etc.).
 5. Para cada timestep, busca la capa más alta (fila mínima) donde aparece el cluster de aerosol:
    - Si esa altura supera 5100 m, exige al menos 7 píxels consecutivos.
    - Si no hay aerosol válido, asigna NaN.
 6. Guarda la serie resultante `time_index aerosol_height_m` en
    `salidastxt/alturamax_cluster_aerosol.txt`.
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

# Grid reducido y timesteps diarios
n_layers, n_times = 32, 64

# Mapping de clusters a categorías
category_mapping = {
    1: 7,  # moléculas
    3: 7,
    5: 1,  # ruido
    4: 5,  # aerosoles
    2: 0,  # nubes
    6: 0,
    0: 1,  # otros
    7: 1
}
AEROSOL_CAT = 5  # categoría de aerosol de superficie

# Duración de cada timestep (~22.5 min)
dt = datetime.timedelta(minutes=24*60/n_times)

# Ruta de salida
txt_path = 'salidastxt'
os.makedirs(txt_path, exist_ok=True)

# Función para extraer fecha del nombre “_AYYYYMMDD.nc”
def extract_date(fname):
    parts = fname.split('_A')
    return datetime.datetime.strptime(parts[1].split('.')[0], '%Y%m%d')

# Carga modelos (una sola vez)
encoder = load_model(encoder_path)
cluster = joblib.load(kmeans_path)

# Acumuladores
all_times = []
all_heights = []
big_grid = np.zeros((n_layers, 0), dtype=int)

# Recorre todos los días
for nc_file in sorted(os.listdir(data_path)):
    if not nc_file.endswith('.nc'):
        continue

    date0 = extract_date(nc_file)
    print(f"Procesando {nc_file}")

    # 1) Altitud
    ds = xr.open_dataset(os.path.join(data_path, nc_file))
    alt = ds['altitude'].values
    ds.close()
    alt = np.flip(np.maximum(alt - station_alt, 0.0), axis=0)
    alt_grid = resize(alt, (n_layers,), order=1,
                      preserve_range=True, anti_aliasing=False)

    # 2) Backscatter → img
    ds = xr.open_dataset(os.path.join(data_path, nc_file))
    rcs = ds['attenuated_backscatter_0'].load().values
    ds.close()
    data = np.log(rcs)
    data = np.flip(data, axis=0)
    data = np.clip(data, -2, 2)
    np.nan_to_num(data, nan=-2, copy=False)
    norm = (data + 2) / 4.0
    img = 1 - norm

    # 3) Encoder + clustering
    img_rs  = resize(img, (256, 512), order=0, anti_aliasing=False)
    xenc    = img_rs[np.newaxis,...,np.newaxis]
    encoded = encoder.predict(xenc)[0]
    flat    = encoded.reshape(-1, encoded.shape[-1])
    lbl_img = cluster.predict(flat).reshape(encoded.shape[:2])

    # 4) Reducir a 32×64 y mapear
    small_lbl = resize(lbl_img, (n_layers, n_times),
                       order=0, preserve_range=True,
                       anti_aliasing=False).astype(int)
    mapped = np.vectorize(lambda x: category_mapping.get(x, -1))(small_lbl)

    # 5) Apéndice horizontal a big_grid
    big_grid = np.concatenate([big_grid, mapped], axis=1)

    # 6) Calcular alturas
    for i in range(n_times):
        tstamp = (date0 + i*dt).strftime('%Y-%m-%d %H:%M')
        rows = np.where(mapped[:, i] == AEROSOL_CAT)[0]
        if rows.size > 0:
            # fila de capa más alta
            hrow = int(rows.min())
            hmax = float(alt_grid[hrow])
            # si está por encima de 5200 m, exigir 5 píxels consecutivos
            if hmax > 5100:
                if not ({hrow, hrow+1, hrow+2, hrow+3, hrow+4, hrow+5, hrow+6 } <= set(rows)):
                    hmax = np.nan
        else:
            hmax = np.nan
        all_times.append(tstamp)
        all_heights.append(hmax)


# 8) Guardar height_series_2024.txt
out_ts = os.path.join(txt_path, 'alturamax_cluster_aerosol.txt')
with open(out_ts, 'w', encoding='utf-8') as f:
    f.write('time_index aerosol_height_m\n')
    for ts, h in zip(all_times, all_heights):
        val = f"{h:.2f}" if not np.isnan(h) else 'nan'
        f.write(f"{ts} {val}\n")
print(f" Serie de alturas guardada en {out_ts}")


