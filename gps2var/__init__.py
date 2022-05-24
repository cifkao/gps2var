import itertools
from typing import Any, Optional, Tuple, Union

import numpy as np
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
        preload_all: Indicates whether the whole dataset should be loaded into memory
            instead of loading one block at a time. Defaults to False.
    """

    def __init__(
        self,
        dataset: rasterio.DatasetReader,
        crs: Any = "EPSG:4326",
        interpolation: str = "nearest",
        block_shape: Optional[Tuple[int, int]] = None,
        preload_all: bool = False,
    ) -> None:
        if interpolation not in ["nearest", "bilinear"]:
            raise ValueError(
                f"interpolation must be 'nearest' or 'bilinear', got {repr(interpolation)}"
            )

        self.dataset = dataset
        self.block_shape = block_shape or dataset.block_shapes[0]
        self.interpolation = interpolation

        self.inv_dataset_transform = ~dataset.transform
        if crs is dataset.crs:
            self.transformer = None
        else:
            self.transformer = pyproj.Transformer.from_crs(
                crs, dataset.crs, always_xy=True
            )
        self.nodata_array = np.asarray(
            [
                np.asarray(val, dtype=dt)
                for dt, val in zip(dataset.dtypes, dataset.nodatavals)
            ]
        )
        self.features_dtype = self.nodata_array.dtype
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
            values = self.at(np.rint(i).astype(int), np.rint(j).astype(int)).astype(
                float
            )
            values[np.isclose(values, self.nodata_array)] = np.nan
            return values

        # round the indices both ways and get the corresponding values
        i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
        i1, j1 = np.ceil(i).astype(int), np.ceil(j).astype(int)
        interp_values = self.at([i0, i0, i1, i1], [j0, j1, j0, j1])

        # compute interpolation weights
        ii = (np.asarray(i) % 1)[..., None]
        jj = (np.asarray(j) % 1)[..., None]
        interp_weights = np.stack(
            [
                np.where(v == self.nodata_array, 0.0, w0 * w1)
                for v, (w0, w1) in zip(
                    interp_values, itertools.product((1 - ii, ii), (1 - jj, jj))
                )
            ]
        )  # shape [4, ..., num_bands]

        assert interp_weights.shape == interp_values.shape

        # renormalize (needed if there are missing values; result is nan if all are missing)
        with np.errstate(invalid="ignore"):
            interp_weights /= interp_weights.sum(axis=0, keepdims=True)

        # bilinear interpolation
        # v00 * (1 - ii) * (1 - jj) + v01 * (1 - ii) * jj + v10 * ii * (1 - jj) + v11 * ii * jj
        return (interp_weights * interp_values).sum(axis=0)

    def at(
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
        result = np.full(
            shape=(*self.features_shape, len(blocks_inverse)),
            fill_value=np.nan,
            dtype=self.features_dtype,
        )
        final_shape = (*row_indices.shape, *self.features_shape)
        row_indices = row_indices.reshape(-1) % block_h
        col_indices = col_indices.reshape(-1) % block_w
        for idx, (i, j) in enumerate(blocks_unique):
            block = self._read_block(i, j)
            (indices,) = np.where(blocks_inverse == idx)
            result[:, indices] = block[:, row_indices[indices], col_indices[indices]]

        return result.T.reshape(final_shape)

    def _read_block(self, i, j):
        block_h, block_w = self.block_shape

        # handle invalid coordinates
        if (
            i < 0
            or i * block_h >= self.dataset.height
            or j < 0
            or j * block_w >= self.dataset.width
        ):
            return self.nodata_block

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
