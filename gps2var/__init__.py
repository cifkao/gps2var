from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyproj
import rasterio


class RasterioValueReader:
    """Enables querying Rasterio dataset using coordinates.

    Args:
        dataset: A Rasterio dataset.
        crs: The coordinate reference system. Defaults to "EPSG:4326", i.e. standard
            GPS coordinates.
        interpolation: "nearest" or "bilinear". Defaults to "nearest".
        block_shape: The shape of the blocks read into memory. Defaults to the block
            shape of the first band of the dataset.
        dtype: The dtype to which to convert the result. Defaults to np.float32.
        fill_value: The value (scalar) with which to replace missing values.
            Defaults to np.nan. If None, the original "nodata" values will be kept.
        preload_all: Indicates whether the whole dataset should be loaded into memory
            instead of loading one block at a time. Defaults to False.
    """

    def __init__(
        self,
        dataset: rasterio.DatasetReader,
        crs: Any = "EPSG:4326",
        interpolation: str = "nearest",
        block_shape: Optional[Tuple[int, int]] = None,
        dtype: npt.DTypeLike = np.float32,
        fill_value: Any = np.nan,
        preload_all: bool = False,
    ) -> None:
        if interpolation not in ["nearest", "bilinear"]:
            raise ValueError(
                f"interpolation must be 'nearest' or 'bilinear', got {repr(interpolation)}"
            )

        self.dataset = dataset
        self.block_shape = block_shape or dataset.block_shapes[0]
        self.interpolation = interpolation
        self.dtype = dtype
        self.fill_value = fill_value

        self.inv_dataset_transform = ~dataset.transform
        if crs == dataset.crs:
            self.transformer = None
        else:
            self.transformer = pyproj.Transformer.from_crs(
                crs, dataset.crs, always_xy=True
            )
        self.nodata_array = np.asarray(
            [
                np.asarray(val, dtype=dt)
                for dt, val in zip(dataset.dtypes, dataset.nodatavals)
            ],
            dtype=dtype,
        )
        self.features_shape = self.nodata_array.shape
        self.nodata_block = np.tile(
            self.nodata_array, (*self.block_shape, 1)
        ).transpose(2, 0, 1)

        self.data = self.dataset.read() if preload_all else None

    def get(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
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
        if self.transformer:
            x, y = self.transformer.transform(x, y)  # transform to dataset coordinnates
        j, i = self.inv_dataset_transform * (x, y)  # transform to (float) pixel indices

        if self.interpolation != "bilinear":
            result = self._at(np.rint(i).astype(int), np.rint(j).astype(int)).astype(
                self.dtype
            )
            self._fill_nodata(result)
            return result

        # round the indices both ways and get the corresponding values
        i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
        i1, j1 = np.ceil(i).astype(int), np.ceil(j).astype(int)
        interp_values = self._at([i0, i0, i1, i1], [j0, j1, j0, j1])
        nodata_mask = np.isclose(interp_values, self.nodata_array)

        # compute interpolation weights
        ii = (np.asarray(i) % 1)[..., None]
        jj = (np.asarray(j) % 1)[..., None]
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
        if self.fill_value is None:
            result = np.where(nodata_mask.all(axis=0), self.nodata_array, result)
        else:
            result[nodata_mask.all(axis=0)] = self.fill_value
        return result

    def at(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ):
        result = self._at(row_indices, col_indices)
        self._fill_nodata(result)
        return result

    def _at(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ):
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
        block_h, block_w = self.block_shape
        blocks = np.stack([row_indices // block_h, col_indices // block_w], axis=-1)
        if blocks.ndim > 1:
            blocks_unique, blocks_inverse = np.unique(
                blocks.reshape(-1, 2), axis=0, return_inverse=True
            )
        else:
            blocks_unique, blocks_inverse = blocks.reshape(1, 2), np.array([0])

        # read the contents of each block, gather the desired values
        result = np.empty(
            shape=(*self.features_shape, len(blocks_inverse)),
            dtype=self.dtype,
        )
        final_shape = (*row_indices.shape, *self.features_shape)
        row_indices = row_indices.reshape(-1) % block_h
        col_indices = col_indices.reshape(-1) % block_w
        for idx, (i, j) in enumerate(blocks_unique):
            block = self._read_block(i, j)
            (indices,) = np.where(blocks_inverse == idx)
            result[:, indices] = block[:, row_indices[indices], col_indices[indices]]

        return result.T.reshape(final_shape)

    def _fill_nodata(self, array):
        if self.fill_value is not None:
            array[np.isclose(array, self.nodata_array)] = self.fill_value

    def _read_block(self, i, j):
        block_h, block_w = self.block_shape

        # handle invalid coordinates
        if (
            i < 0
            or i * block_h >= self.dataset.height
            or j < 0
            or j * block_w >= self.dataset.width
        ):
            return self.nodata_block.copy()

        i0, i1 = block_h * i, block_h * (i + 1)
        j0, j1 = block_w * j, block_w * (j + 1)

        if self.data is not None:
            data = self.data[:, i0:i1, j0:j1]
        else:
            data = self.dataset.read(window=((i0, i1), (j0, j1)))

        # handle incomplete blocks at the boundary
        if data.shape[1:] != (block_h, block_w):
            padded = self.nodata_block.copy()
            padded[:, : data.shape[1], : data.shape[2]] = data
            data = padded

        return data
