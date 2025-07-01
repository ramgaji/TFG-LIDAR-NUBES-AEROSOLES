import xarray as xr
from pathlib import Path
from rich.progress import track
import matplotlib.pyplot as plt
import numpy as np

path = 'validation'
yyyy = '2024'
d1, d2 = 2, 2
m1, m2 = 1, 11
mms = [str(m).zfill(2) for m in list(range(m1,m2+1))]
dds = [str(d).zfill(2) for d in list(range(d1,d2+1))]

# list all AP files for a given date
files = [
    [f for f in Path('..', 'data', yyyy, mm, dd).iterdir() if f.is_file()]
    for mm in mms
    for dd in dds
]
files = np.concatenate(files, axis=0)

for file in track(files):
    var = 'attenuated_backscatter_0'
    rcs = xr.open_dataset(file)[var].load()
    # filter wavelength: 1064 nm intstruments shoudl be calibrated
    if '1064 nm' not in rcs.attrs['long_name']:
        continue
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 6))

    # Plot attenuated_backscatter_0
    try:
        np.log(rcs).transpose().plot( 
            vmin=-2,
            vmax=2,
            cmap='gray_r', #'coolwarm',
            add_colorbar=False
        )
        # Adjust layout for better spacing
        plt.axis('off')
        Path('images',path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path('images', path, f'{file.stem}.png'), bbox_inches='tight', pad_inches = 0)
        plt.close()
    except:
        continue