# gps2var
Fast loading of geospatial variables from raster files by GPS coordinates and with interpolation.

```python
PATH = "/vsizip/wc2.1_30s_bio.zip/wc2.1_30s_bio_1.tif"  # WorldClim annual mean temperature

with gps2var.RasterValueReader(PATH, interpolation='bilinear') as reader:
    lat, lon = 48.858222, 2.2945
    reader.get(lon, lat)  # [11.94036207]
```

## Parallel reading from multiple rasters
`MultiRasterValueReader` reads from multiple raster files and concatenates the results.

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

Set `use_multiprocessing=True` to create a separate process for each raster. This is likely to be faster than the default (i.e. multithreading), at least if the number of rasters is large.

## PyTorch DataLoaders
Using a `RasterValueReader` with a PyTorch DataLoader with `num_workers > 0` and with the `"fork"` [start method](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods) (default on Unix) will not work.

Here are examples of usage that do work:
- Using `multiprocessing.set_start_method("spawn")`. This will create a copy of the reader in each worker process.
- Setting `preload_all=True` so that the whole raster is loaded into memory.
- (with multiple rasters) Using `MultiRasterValueReader` as above, but setting `use_multiprocessing=True`. This way, each raster wil be read in a separate process.
- Using `ProcessManager`, e.g.:

  ```python
  # in __init__:
  self.manager = gps2var.ProcessManager()
  self.manager.start()  # start a new thread
  self.reader = manager.RasterValueReader()  # proxy object
  
  # in __getitem__:
  self.reader.get(lon, lat)
  ```
  
  In this case, the reader is placed in a separate process and the workers connect to it using the proxy object.
