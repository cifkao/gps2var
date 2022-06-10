# gps2var
`gps2var` is a Python library providing fast loading of geospatial variables from raster files by GPS coordinates and with interpolation.

In particular, it allows parallel calls with coordinates specified as large NumPy arrays of arbitrary shapes, and is compatible with PyTorch data loaders.

Install with: `pip install gps2var`

See [`benchmark.ipynb`](https://github.com/cifkao/gps2var/blob/main/benchmark.ipynb) for a performance benchmark.

## Examples
### Reading from a single file
```python
PATH = "/vsizip/wc2.1_30s_bio.zip/wc2.1_30s_bio_1.tif"  # WorldClim annual mean temperature

with gps2var.RasterValueReader(PATH, interpolation='bilinear') as reader:
    lat, lon = 48.858222, 2.2945
    reader.get(lon, lat)  # [11.94036207]
```

### Parallel reading from multiple files
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

## API

### `RasterValueReader`

Can be created with a path to a file that can be read by Rasterio, or an open Rasterio `DatasetReader`. The behavior can be customized with keyword parameters; the most important ones are:
- `crs`: The coordinate reference system to use for querying. By default this is EPSG:4326, i.e. longitude and latitude (in this order) as used by GPS.
- `interpolation`: `"nearest"` (default) or `"bilinear"` (slower).
- `fill_value`: The value (scalar) with which to replace missing values. Defaults to `np.nan`.
- `feat_dtype`: The dtype to which to convert the result. Defaults to `np.float32`.
- `feat_center`: Center each of the features at the given value. Defaults to `None` (no centering).
- `feat_scale`: Scale each of the (centered) features by multiplying it by the given value. Defaults to `None` (no scaling).
- `block_shape`: The shape of the blocks read into memory (and stored in the GDAL block cache).
- `preload_all`: Indicates whether the whole dataset should be loaded into memory instead of loading one block at a time. Defaults to `False`.

Another option is to wrap all these arguments in a `RasterReaderSpec` (or simply a dictionary) and pass it as the first argument.

### `RasterValueReaderPool`

Like `RasterValueReader`, but spawns `num_workers` worker processes that all read from the same file concurrently.
Besides `get()` (which blocks until the result is ready), it provides `async_get()`, which returns a `concurrent.futures.Future`.

### `MultiRasterValueReader`

Expects as the first argument a list of file paths, `RasterReaderSpec`s, or `dict`s, and reads from each file in a separate thread or process. Additional options to be applied to all items can be passed as keyword arguments. Additionally, the following parameters are accepted:
- `use_multiprocessing`: If `True`, each raster will be processed in a separate process. 
- `num_threads`: The number of threads to use for parallel reading. By default, this is set to the number of rasters. Set to 0 to read in the main thread.

### `ProcessManager`

A `multiprocessing.BaseManager` – a context manager that spawns a separate process. It provides `RasterValueReader()`, `MultiRasterValueReader()`, and `RasterValueReaderPool()` methods that create the corresponding reader in that process and return a proxy object that can be used in much the same way as the reader itself. A nice property of a proxy object is that it can be copied between processes without copying the underlying reader, so it works with PyTorch `DataLoader`.

## PyTorch `DataLoader` and parallelism
Simply using a `RasterValueReader` with a PyTorch `DataLoader` with `num_workers > 0` and with the `"fork"` [start method](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods) (default on Unix) **will not work**.

Here are examples of usage that do work:
- Using `multiprocessing.set_start_method("spawn")`. This will create a copy of the reader in each worker process.
- Setting `preload_all=True` so that the whole raster is loaded into memory.
- Using `MultiRasterValueReader` as above, but with `use_multiprocessing=True`. This way, each raster wil be read in a separate process.
- Using `ProcessManager`, e.g.:

  ```python
  # in __init__:
  self.manager = gps2var.ProcessManager()
  self.manager.start()  # start a new process
  self.reader = manager.RasterValueReader(path)  # proxy object
  
  # in __getitem__:
  self.reader.get(lon, lat)
  ```
  
  In this case, the reader is placed in a separate process and the workers connect to it using the proxy object.

## Caveats and limitations

- `gps2var` reads the raster in windows, which is efficient _if the locations requested in a single call tend to be close together_. If this is not the case, better performance can be achieved using `preload_all=True` at the expense of loading the whole raster into memory.
- By default, the window shape used by `gps2var` is identical to the shape of the blocks in which the dataset is stored. This ensures the windows can be read efficiently, but it might mean reading a lot of data unnecessarily if the block size is large. Adjusting the window shape using the `block_shape` paramater might improve performance.
