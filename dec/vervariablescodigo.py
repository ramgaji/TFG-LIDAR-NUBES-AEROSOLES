import os
import xarray as xr


nc_file = os.path.join('..', 'data', 'granada', '2025', 'L2_0-20008-0-UGR_A20250104.nc')
if not os.path.exists(nc_file):
    print("El archivo no existe:", nc_file)
else:
    ds = xr.open_dataset(nc_file)
    print(ds)


ds = xr.open_dataset(nc_file)

# 1) Imprimir un resumen de todo el dataset
print(ds)

# 2) Listar las variables disponibles
print("Variables en el dataset:", list(ds.variables))

# 3) Inspeccionar en detalle una variable concreta
var_name = "cloud_base_height"  
print(ds[var_name])

# 4) Acceder a atributos globales
print(ds.attrs)

# 5) Cerrar el dataset (opcional si usas context manager o lazy loading)
ds.close()





