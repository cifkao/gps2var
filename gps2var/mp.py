import concurrent.futures as cf
import contextlib
from multiprocessing.managers import BaseManager
from typing import Any, List, Optional, Union

import numpy as np

from . import core


class Manager(BaseManager):
    pass


Manager.register("RasterValueReader", core.RasterValueReader)


class MultiRasterValueReader(core.RasterValueReaderBase):
    def __init__(
        self,
        specs: List[core.RasterReaderSpecLike],
        num_workers: Optional[int] = None,
        **kwargs
    ):
        self._managers_stack = contextlib.ExitStack()
        self.readers = []
        for spec in specs:
            manager = self._managers_stack.enter_context(Manager())
            proxy = manager.RasterValueReader(spec=spec, **kwargs)
            self.readers.append(proxy)

        self._pool = cf.ThreadPoolExecutor(num_workers) if num_workers else None
        self._map_fn = self._pool.map if self._pool else map

    def close(self):
        try:
            if self._pool:
                self._pool.shutdown()
        finally:
            self._managers_stack.close()

    def get(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        extra: Any = None,
    ) -> np.ndarray:
        results = self._map_fn(lambda rd: rd.get(x, y, extra), self.readers)
        return np.concatenate(list(results), axis=-1)

    def at(
        self, row_indices: Union[int, np.ndarray], col_indices: Union[int, np.ndarray]
    ) -> np.ndarray:
        results = self._map_fn(lambda rd: rd.at(row_indices, col_indices), self.readers)
        return np.concatenate(list(results), axis=-1)
