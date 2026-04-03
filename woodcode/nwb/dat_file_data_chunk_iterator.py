import numpy as np
from tqdm import tqdm

from neuroconv.tools.hdmf import GenericDataChunkIterator


class DatFileDataChunkIterator(GenericDataChunkIterator):
    """DataChunkIterator for lazily reading a Neuroscope .dat file via a pynapple TsdFrame.

    Avoids loading the full file into memory by slicing one chunk at a time in the time
    dimension before applying channel reordering.
    """

    def __init__(
        self,
        raw_data,
        chan_order: np.ndarray,
        buffer_gb: float | None = None,
        buffer_shape: tuple | None = None,
        chunk_mb: float | None = None,
        chunk_shape: tuple | None = None,
        display_progress: bool = False,
        progress_bar_class: tqdm | None = None,
        progress_bar_options: dict | None = None,
    ):
        self._raw_data = raw_data
        self._chan_order = chan_order

        super().__init__(
            buffer_gb=buffer_gb,
            buffer_shape=buffer_shape,
            chunk_mb=chunk_mb,
            chunk_shape=chunk_shape,
            display_progress=display_progress,
            progress_bar_class=progress_bar_class,
            progress_bar_options=progress_bar_options,
        )

    def _get_data(self, selection: tuple[slice]) -> np.ndarray:
        time_slice = selection[0]
        start = time_slice.start if time_slice.start is not None else 0
        stop = time_slice.stop if time_slice.stop is not None else self._get_maxshape()[0]

        # Load only this time chunk from the memory-mapped file
        chunk = np.array(self._raw_data[start:stop, :])

        # Apply channel reordering, then any channel sub-selection from the iterator
        chunk = chunk[:, self._chan_order]
        chunk = chunk[:, selection[1]]
        return chunk

    def _get_dtype(self) -> np.dtype:
        return np.dtype("int16")

    def _get_maxshape(self) -> tuple[int, int]:
        return (len(self._raw_data), len(self._chan_order))
