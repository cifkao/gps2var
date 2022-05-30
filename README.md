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
