# gps2var
Fast loading of geospatial variables from raster files by GPS coordinates and with interpolation.

```python
import rasterio
from gps2var import RasterioValueReader

dataset = rasterio.open("wildareas-v3-2009-human-footprint.tif")
value_reader = RasterioValueReader(dataset=dataset, interpolation="bilinear")

lat, lon = 48.858222, 2.2945
value_reader.get(lon, lat)  # array([36.72506563])
```

## Multiprocessing
The Rasterio dataset cannot be shared by multiple processes. When using multiprocessing (e.g. in a PyTorch `DataLoader`), either make sure to open the dataset anew in each worker process, or pass `preload_all=True` to load the whole dataset into memory and share it across workers.
