# gps2var
Fast loading of geospatial variables from raster files by GPS coordinates and with interpolation.

```python
PATH = "/vsizip/wc2.1_30s_bio.zip/wc2.1_30s_bio_1.tif"  # WorldClim annual mean temperature

with gps2var.RasterValueReader(PATH, interpolation='bilinear') as reader:
    lat, lon = 48.858222, 2.2945
    reader.get(lon, lat)  # [11.94036207]
```

## Parallel reading from multiple rasters
`MultiRasterValueReader` reads from multiple datasets and concatenates the results.

```python
# min and max temperature by month (averages for 1970-2000)
PATHS = [f"/vsizip/wc2.1_30s_{var}.zip/wc2.1_30s_{var}_{i:02}.tif"
         for var in ["tmin" , "tmax"] for i in range(1, 13)]

with gps2var.MultiRasterValueReader(PATHS, num_threads=len(PATHS)) as reader:
    lat, lon = 48.858222, 2.2945
    reader.get(lon, lat).reshape(2, 12)
    
# [[ 2.3  2.5  4.6  6.3 10.1 13.  15.  14.9 12.   8.8  5.   3.4]
#  [ 7.2  8.4 11.9 14.9 19.2 22.  24.7 24.8 20.9 15.9 10.6  8. ]]
```

Use `MultiRasterValueReader` with `use_multiprocessing=True` to create a separate process for each dataset.
