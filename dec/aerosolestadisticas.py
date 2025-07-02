"""
aerosolestadisticas.py

Carga la serie `alturamax_cluster_aerosol.txt` (tiempo × altura máxima de aerosol),
genera un gráfico anual de la altura, define una función para extraer el máximo
corroborado en rachas ≥3 valores iguales y calcula estadísticas por estación
(e.g. media, desviación, error de la media, pico corroborado) tanto globales
como separadas día/noche. Finalmente guarda todas las tablas en
`salidastxt/estadisticasaerosol.txt`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ————— Ruta de la carpeta de entrada —————
txt_path = 'salidastxt'
os.makedirs(txt_path, exist_ok=True)



# ————— Archivo de serie de alturas —————
height_file = os.path.join(txt_path, 'alturamax_cluster_aerosol.txt')

# ————— 1) Leer manualmente las dos columnas —————
times = []
heights = []
with open(height_file, 'r', encoding='utf-8') as f:
    next(f)  # saltar la cabecera
    for line in f:
        time_str = line[:16]                     # "YYYY-MM-DD HH:MM"
        h_str    = line[16:].strip()
        times.append(pd.to_datetime(time_str, format='%Y-%m-%d %H:%M'))
        heights.append(float(h_str) if h_str.lower()!='nan' else np.nan)

df = pd.DataFrame({'aerosol_height_m': heights}, index=times)

# ————— 2) Generar plot anual —————
plt.figure(figsize=(12,4))
plt.plot(
    df.index,
    df['aerosol_height_m'],
    marker='s',            # cuadrado
    linestyle='None',      # sin líneas
    markersize=2          # tamaño grande
)
plt.ylabel('Altura aerosol (m)')
plt.xlabel('Fecha')
plt.title('Serie anual altura máxima de aerosol')
plt.tight_layout()
out_png = os.path.join(txt_path, 'height_series_2024.png')
plt.savefig(out_png)
plt.close()
print(f'→ Plot guardado en {out_png}')

# ————— función para altura máxima corroborada —————
def max_corroborated(arr):
    """Devuelve el máximo valor que forma una racha >=3 del mismo valor."""
    if len(arr) < 3:
        return np.nan
    best = np.nan
    curr_val = arr[0]
    curr_len = 1
    runs = []
    for v in arr[1:]:
        if np.isnan(v) or np.isnan(curr_val):
            runs.append((curr_val, curr_len))
            curr_val = v
            curr_len = 1
        elif v == curr_val:
            curr_len += 1
        else:
            runs.append((curr_val, curr_len))
            curr_val = v
            curr_len = 1
    runs.append((curr_val, curr_len))
    for val, length in runs:
        if not np.isnan(val) and length >= 3:
            best = val if np.isnan(best) else max(best, val)
    return best

# ————— 3) Estadísticas por estación —————
seasons = {
    'Invierno': (12,1,2),
    'Primavera': (3,4,5),
    'Verano': (6,7,8),
    'Otoño': (9,10,11)
}
season_stats = []
for season, months in seasons.items():
    sub = df[df.index.month.isin(months)]['aerosol_height_m']
    total = len(sub)
    n_nan = sub.isna().sum()
    count = sub.count()
    mean_h = sub.mean(skipna=True)
    std_h  = sub.std(skipna=True)
    sem_h  = std_h / np.sqrt(count) if count > 0 else np.nan
    max_corr = max_corroborated(sub.values)
    season_stats.append((season, total, n_nan, mean_h, std_h, sem_h, max_corr))

# ————— 4) Estadísticas día/noche por estación —————
dn_stats = []
for season, months in seasons.items():
    sub = df[df.index.month.isin(months)]
    # día
    day = sub.between_time('06:00','17:59')['aerosol_height_m']
    total_d = len(day)
    n_nan_d = day.isna().sum()
    count_d = day.count()
    mean_d  = day.mean(skipna=True)
    std_d   = day.std(skipna=True)
    sem_d   = std_d / np.sqrt(count_d) if count_d > 0 else np.nan
    max_d   = max_corroborated(day.values)
    dn_stats.append((season, 'Día', total_d, n_nan_d, mean_d, std_d, sem_d, max_d))
    # noche
    night = sub.drop(day.index)['aerosol_height_m']
    total_n = len(night)
    n_nan_n = night.isna().sum()
    count_n = night.count()
    mean_n  = night.mean(skipna=True)
    std_n   = night.std(skipna=True)
    sem_n   = std_n / np.sqrt(count_n) if count_n > 0 else np.nan
    max_n   = max_corroborated(night.values)
    dn_stats.append((season, 'Noche', total_n, n_nan_n, mean_n, std_n, sem_n, max_n))

# ————— 5) Escribir estadísticas en un TXT —————
out_stats = os.path.join(txt_path, 'estadisticasaerosol.txt')
with open(out_stats, 'w', encoding='utf-8') as f:
    # encabezado estación
    f.write('Estación  Total   NaNs  AlturaMaxMedia(m)  DesvStd(m)  ErrMedia(m)  AlturaMaxPico(m)\n')
    for season, total, n_nan, mean_h, std_h, sem_h, max_c in season_stats:
        f.write(f'{season:10s} {total:6d} {n_nan:6d} ' +
                f'{mean_h:16.2f} {std_h:11.2f} {sem_h:12.2f} {max_c:18.2f}\n')
    f.write('\n')
    # encabezado día/noche
    f.write('Estación  Período  Total   NaNs  AlturaMaxMedia(m)  DesvStd(m)  ErrMedia(m)  AlturaMaxPico(m)\n')
    for season, period, total, n_nan, mean_h, std_h, sem_h, max_c in dn_stats:
        f.write(f'{season:10s} {period:7s} {total:6d} {n_nan:6d} ' +
                f'{mean_h:16.2f} {std_h:11.2f} {sem_h:12.2f} {max_c:18.2f}\n')

print(f'→ Estadísticas (con desviación estándar y error de media) guardadas en {out_stats}')





