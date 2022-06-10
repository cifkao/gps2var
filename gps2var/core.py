import abc
import concurrent.futures as cf
import contextlib
import dataclasses
from multiprocessing.managers import BaseManager, BaseProxy
import os
import threading
from typing import Any, List, Optional, Iterable, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyproj
import rasterio, rasterio.io, rasterio.path


class RasterValueReaderBase(abc.ABC):
    @abc.abstractmethod
    def get(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        extra: Any = None,
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclasses.dataclass(frozen=True)
class RasterReaderSpec:
    """An object specifying how a dataset should be read.

    Attributes:
        path: A path to a Rasterio dataset.
        open_options: Keyword arguments for rasterio.open().
        crs: The coordinate reference system. Defaults to "EPSG:4326", i.e. standard
            GPS coordinates.
        interpolation: "nearest" or "bilinear". Defaults to "nearest".
        block_shape: The shape of the blocks read into memory. Defaults to the block
            shape of the first band of the dataset.
        fill_value: The value (scalar) with which to replace missing values.
            Defaults to np.nan. If None, the original "nodata" values will be kept.
        feat_dtype: The dtype to which to convert the result. Defaults to np.float32.
        feat_center: Center each of the features at the given value. Defaults to None
            (no centering).
        feat_scale: Scale each of the (centered) features by multiplying it by the
            given value. Defaults to None (no scaling).
        preload_all: Indicates whether the whole dataset should be loaded into memory
            instead of loading one block at a time. Defaults to False.
        early_cast: Indicates whether the data should be cast to feat_dtype as soon as
            it is read. Defaults to True.
    """

    path: Union[str, os.PathLike, rasterio.path.Path]
    open_options: dict = dataclasses.field(default_factory=dict)
    crs: Any = "EPSG:4326"
    interpolation: str = "nearest"
    block_shape: Optional[Tuple[int, int]] = None
    fill_value: Any = np.nan
    feat_dtype: npt.DTypeLike = np.float32
    feat_center: Optional[Union[Any, List[Any]]] = None
    feat_scale: Optional[Union[Any, List[Any]]] = None
    preload_all: bool = False
    early_cast: bool = True

    def __post_init__(self):
        if self.interpolation not in ["nearest", "bilinear"]:
            raise ValueError(
                f"interpolation must be 'nearest' or 'bilinear', got {repr(self.interpolation)}"
            )

    @classmethod
    def from_any(
        cls, obj: Optional["RasterReaderSpecLike"], **kwargs
    ) -> "RasterReaderSpec":
        if obj is None:
            return cls(**kwargs)
        if isinstance(obj, cls):
            if kwargs:
                return dataclasses.replace(obj, **kwargs)
            return obj
        if isinstance(obj, dict):
            return cls(**{**obj, **kwargs})
        return cls(path=obj, **kwargs)


RasterReaderSpecLike = Union[
    RasterReaderSpec, dict, str, os.PathLike, rasterio.path.Path
]


class RasterValueReader(RasterValueReaderBase):
    """Enables querying a Rasterio dataset using coordinates."""

    def __init__(
        self,
        spec: Optional[RasterReaderSpecLike] = None,
        dataset: Optional[rasterio.DatasetReader] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self._created_from_dataset = dataset is not None

        if self._created_from_dataset and "path" not in kwargs:
            kwargs["path"] = dataset.name
        spec = RasterReaderSpec.from_any(spec, **kwargs)
        self.spec = spec

        with contextlib.ExitStack() as ctx:
            if dataset is None:
                dataset = rasterio.open(spec.path, **spec.open_options)
                if spec.preload_all:
                    ctx.enter_context(dataset)

            self._feat_center, self._feat_scale = spec.feat_center, spec.feat_scale

            if self._feat_center is not None:
                if not isinstance(self._feat_center, Iterable):
                    self._feat_center = [self._feat_center] * dataset.count
                if len(self._feat_center) != dataset.count:
                    raise ValueError(
                        f"Expected feat_center to have {dataset.count} elements, "
                        f"got {len(self._feat_center)}"
                    )
                self._feat_center = np.asarray(self._feat_center)

            if self._feat_scale is not None:
                if not isinstance(self._feat_scale, Iterable):
                    self._feat_scale = [self._feat_scale] * dataset.count
                if len(self._feat_scale) != dataset.count:
                    raise ValueError(
                        f"Expected feat_scale to have {dataset.count} elements, "
                        f"got {len(self._feat_scale)}"
                    )
                self._feat_scale = np.asarray(self._feat_scale)

            if not spec.preload_all:
                self._block_shape = spec.block_shape or dataset.block_shapes[0]
            else:
                # use one single block since the whole dataset will be loaded anyway
                self._block_shape = dataset.shape
            self._inv_dataset_transform = ~dataset.transform
            if spec.crs == dataset.crs:
                self._transformer = None
            else:
                self._transformer = pyproj.Transformer.from_crs(
                    spec.crs, dataset.crs, always_xy=True
                )
            self._nodata_array = np.asarray(dataset.nodatavals, dtype=spec.feat_dtype)
            self._feat_shape = self._nodata_array.shape
            self._nodata_block = np.tile(
                self._nodata_array, (*self._block_shape, 1)
            ).transpose(2, 0, 1)

            self._read_lock = threading.Lock()
            if spec.preload_all:
                with self._read_lock:
                    self.data = dataset.read(
                        out_dtype=spec.feat_dtype if spec.early_cast else None
                    )
            else:
                self.data = None

            self.dataset = dataset

    def close(self):
        self.dataset.close()

    def get(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        extra: Any = None,
    ) -> np.ndarray:
        """Read values from the dataset by coordinates.

        Note that for EPSG:4326 (default), the coordinates need to be passed as
        x=longitude, y=latitude.

        Args:
            x: A single coordinate or an array of coordinates of any shape.
            y: A single coordinate or an array of coordinates of the same shape as x.

        Returns:
            A NumPy array of shape (*x.shape, dataset.count).
        """
        x, y = np.asarray(x), np.asarray(y)
        if self._transformer:
            # transform to dataset coordinnates
            x, y = self._transformer.transform(x, y)
        # transform to (float) pixel indices
        j, i = self._inv_dataset_transform * (x, y)

        if self.spec.interpolation != "bilinear":
            result = self._iget(np.rint(i).astype(int), np.rint(j).astype(int))
            self._fill_nodata(result)
            self._normalize(result)
            return result

        # round the indices both ways and get the corresponding values
        i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
        i1, j1 = np.ceil(i).astype(int), np.ceil(j).astype(int)
        interp_values = self._iget([i0, i0, i1, i1], [j0, j1, j0, j1])
        nodata_mask = np.isclose(interp_values, self._nodata_array)

        # compute interpolation weights
        w_dtype = interp_values.dtype
        if w_dtype.kind != "f":
            w_dtype = np.float_
        ii = (np.asarray(i) % 1)[..., None].astype(w_dtype)
        jj = (np.asarray(j) % 1)[..., None].astype(w_dtype)
        interp_weights = np.stack(
            [(1 - ii) * (1 - jj), (1 - ii) * jj, ii * (1 - jj), ii * jj]
        )  # shape [4, ..., num_bands]
        interp_weights[nodata_mask] = 0.0

        assert interp_weights.shape == interp_values.shape

        # renormalize (needed if there are missing values)
        with np.errstate(invalid="ignore", divide="ignore"):
            interp_weights /= interp_weights.sum(axis=0, keepdims=True)

        # bilinear interpolation
        # v00 * (1 - ii) * (1 - jj) + v01 * (1 - ii) * jj + v10 * ii * (1 - jj) + v11 * ii * jj
        result = (interp_weights * interp_values).sum(axis=0)
        if self.spec.fill_value is None:
            result = np.where(nodata_mask.all(axis=0), self._nodata_array, result)
        else:
            result[nodata_mask.all(axis=0)] = self.spec.fill_value
        self._normalize(result)
        return result

    def iget(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ) -> np.ndarray:
        result = self._iget(row_indices, col_indices)
        self._fill_nodata(result)
        self._normalize(result)
        return result

    def _iget(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ) -> np.ndarray:
        """Read values from the dataset by row and column indices.

        Args:
            row_indices: A single row index or an array of row indices of any shape.
            col_indices: A single column index or an array of column indices of the
                same shape as row_indices.

        Returns:
            A NumPy array of shape (*row_indices.shape, dataset.count).
        """
        row_indices, col_indices = np.asarray(row_indices), np.asarray(col_indices)
        # find which blocks we need to load
        block_h, block_w = self._block_shape
        blocks = np.stack([row_indices // block_h, col_indices // block_w], axis=-1)
        if blocks.ndim > 1:
            blocks_unique, blocks_inverse = np.unique(
                blocks.reshape(-1, 2), axis=0, return_inverse=True
            )
        else:
            blocks_unique, blocks_inverse = blocks.reshape(1, 2), np.array([0])

        # read the contents of each block, gather the desired values
        result = np.empty(
            shape=(*self._feat_shape, len(blocks_inverse)),
            dtype=self.spec.feat_dtype,
        )
        final_shape = (*row_indices.shape, *self._feat_shape)
        row_indices = row_indices.reshape(-1) % block_h
        col_indices = col_indices.reshape(-1) % block_w
        for idx, (i, j) in enumerate(blocks_unique):
            block = self._read_block(i, j)
            (indices,) = np.where(blocks_inverse == idx)
            result[:, indices] = block[:, row_indices[indices], col_indices[indices]]

        return result.T.reshape(final_shape)

    def _fill_nodata(self, array: np.ndarray) -> None:
        if self.spec.fill_value is not None:
            array[np.isclose(array, self._nodata_array)] = self.spec.fill_value

    def _normalize(self, array: np.ndarray) -> None:
        with np.errstate(invalid="ignore"):
            if self._feat_center is not None:
                array -= self._feat_center
            if self._feat_scale is not None:
                array *= self._feat_scale

    def _read_block(self, i: int, j: int) -> np.ndarray:
        block_h, block_w = self._block_shape

        # handle invalid coordinates
        if (
            i < 0
            or i * block_h >= self.dataset.height
            or j < 0
            or j * block_w >= self.dataset.width
        ):
            return self._nodata_block.copy()

        i0, i1 = block_h * i, block_h * (i + 1)
        j0, j1 = block_w * j, block_w * (j + 1)

        if self.data is not None:
            data = self.data[:, i0:i1, j0:j1]
        else:
            with self._read_lock:
                data = self.dataset.read(
                    window=((i0, i1), (j0, j1)),
                    out_dtype=self.spec.feat_dtype if self.spec.early_cast else None,
                )

        # handle incomplete blocks at the boundary
        if data.shape[1:] != (block_h, block_w):
            padded = self._nodata_block.copy()
            padded[:, : data.shape[1], : data.shape[2]] = data
            data = padded

        return data

    def __reduce__(self):
        if self._created_from_dataset:
            raise RuntimeError(
                "Cannot serialize a reader created from a "
                "rasterio.DatasetReader object"
            )
        return (self._reconstruct, (self.spec,))

    @classmethod
    def _reconstruct(cls, spec):
        return cls(spec=spec)


class MultiRasterValueReader(RasterValueReaderBase):
    """A convenience reader that reads from multiple readers and concatentates the
    results.
    """

    _reduce_kwargs = ["readers", "num_threads", "use_multiprocessing"]

    def __init__(
        self,
        specs: Optional[List[RasterReaderSpecLike]] = None,
        datasets: Optional[List[rasterio.DatasetReader]] = None,
        readers: Optional[List[Union[RasterValueReaderBase, BaseProxy]]] = None,
        num_threads: Optional[int] = None,
        use_multiprocessing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self._exit_stack = contextlib.ExitStack()
        try:
            if readers is None:
                if specs is None and datasets is None:
                    raise TypeError(
                        "At least one of specs, datasets and readers must be given"
                    )
                specs = specs or [None] * len(datasets)
                datasets = datasets or [None] * len(specs)
                if len(specs) != len(datasets):
                    raise ValueError("specs and datasets must be the same length")

                if use_multiprocessing:
                    # Create a separate process for each dataset
                    managers = [
                        self._exit_stack.enter_context(ProcessManager()) for _ in specs
                    ]
                else:
                    managers = [None] * len(specs)

                def make_reader(spec, dataset, manager=None):
                    if manager:
                        return contextlib.closing(
                            manager.RasterValueReader(
                                spec=spec, dataset=dataset, **kwargs
                            )
                        )
                    return RasterValueReader(spec=spec, dataset=dataset, **kwargs)

                if num_threads is None:
                    num_threads = len(specs)

                # Create readers, possibly in parallel
                with _parallel_map_fn(num_threads) as map_fn:
                    self.readers = [
                        self._exit_stack.enter_context(reader)
                        for reader in map_fn(make_reader, specs, datasets, managers)
                    ]
            else:
                if not (specs is None and datasets is None and not kwargs):
                    raise TypeError(
                        "Unexpected specs, datasets or kwargs because "
                        "readers were given"
                    )
                self.readers = readers

            if num_threads is None:
                num_threads = len(self.readers)

            # Making a new thread pool here that we only use after forking (if using
            # multiprocessing), otherwise we could get a deadlock
            self._map_fn = self._exit_stack.enter_context(_parallel_map_fn(num_threads))

            self.num_threads = num_threads
            self.use_multiprocessing = use_multiprocessing
        except:
            with self._exit_stack:
                raise

    def close(self):
        self._exit_stack.close()

    def get(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        extra: Any = None,
    ) -> np.ndarray:
        results = self._map_fn(lambda rd: rd.get(x, y, extra), self.readers)
        return np.concatenate(list(results), axis=-1)

    def iget(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ) -> np.ndarray:
        results = self._map_fn(
            lambda rd: rd.iget(row_indices, col_indices), self.readers
        )
        return np.concatenate(list(results), axis=-1)

    def __reduce__(self):
        return (
            self._reconstruct,
            ({k: self.__dict__[k] for k in self._reduce_kwargs},),
        )

    @classmethod
    def _reconstruct(cls, kwargs):
        return cls(**kwargs)


class RasterValueReaderPool(RasterValueReaderBase):
    """A pool of readers reading all in parallel from the same file."""

    def __init__(
        self,
        spec: RasterReaderSpecLike,
        num_workers: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self._exit_stack = contextlib.ExitStack()
        try:

            def initializer(spec, kwargs):
                global _WORKER_DATA
                _WORKER_DATA = {"reader": RasterValueReader(spec=spec, **kwargs)}

            self._pool = self._exit_stack.enter_context(
                cf.ProcessPoolExecutor(
                    num_workers, initializer=initializer, initargs=(spec, kwargs)
                )
            )
        except:
            with self._exit_stack:
                raise

    def close(self):
        self._exit_stack.close()

    def async_get(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        extra: Any = None,
    ) -> cf.Future:
        return self._pool.submit(self._worker_get, x, y, extra)

    def get(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        extra: Any = None,
    ) -> np.ndarray:
        return self.async_get(x, y, extra).result()

    @staticmethod
    def _worker_get(*args):
        return _WORKER_DATA["reader"].get(*args)

    def async_iget(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ) -> cf.Future:
        return self._pool.submit(self._worker_iget, row_indices, col_indices)

    def iget(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ) -> np.ndarray:
        return self.async_iget(row_indices, col_indices).result()

    @staticmethod
    def _worker_iget(*args):
        return _WORKER_DATA["reader"].iget(*args)


class ZipRasterValueReader(MultiRasterValueReader):
    def __init__(
        self,
        zip_path: str,
        specs: List[RasterReaderSpecLike],
        num_threads: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._zip_file = open(zip_path, "rb")
        self._zip_memory_file = rasterio.io.ZipMemoryFile(self._zip_file)
        datasets = [
            self._zip_memory_file.open(spec.path, **spec.open_options) for spec in specs
        ]

        super().__init__(
            specs=specs, datasets=datasets, num_threads=num_threads, **kwargs
        )

        if kwargs.get("preload_all", False) or all(sp.preload_all for sp in specs):
            self._zip_memory_file.close()
            self._zip_file.close()
            self._zip_memory_file, self._zip_file = None, None

    def close(self):
        try:
            for reader in self.readers:
                reader.close()
        finally:
            if self._zip_file:
                self._zip_memory_file.close()
                self._zip_file.close()


@contextlib.contextmanager
def _parallel_map_fn(num_threads):
    if num_threads:
        with cf.ThreadPoolExecutor(num_threads) as pool:
            yield pool.map
    else:
        yield map


_WORKER_DATA = None  # process-local storage for process pools


class ProcessManager(BaseManager):
    pass


ProcessManager.register("RasterValueReader", RasterValueReader)
ProcessManager.register("MultiRasterValueReader", MultiRasterValueReader)
ProcessManager.register("RasterValueReaderPool", RasterValueReaderPool)
ProcessManager.register("ZipRasterValueReader", ZipRasterValueReader)
