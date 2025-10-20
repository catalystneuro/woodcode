from tqdm import tqdm
import numpy as np
from spikeinterface import BaseRecording

from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)
from neuroconv.tools.hdmf import GenericDataChunkIterator

class MultiSegmentRecordingDataChunkIterator(GenericDataChunkIterator):
    """DataChunkIterator specifically for use on a multi-segment recording.

    This class concatenates multiple SpikeInterfaceRecordingDataChunkIterator
    instances, each corresponding to a segment in the multi-segment recording.
    """

    def __init__(
        self,
        recording: BaseRecording,
        segment_indices: list[int],
        buffer_gb: float | None = None,
        buffer_shape: tuple | None = None,
        chunk_mb: float | None = None,
        chunk_shape: tuple | None = None,
        display_progress: bool = False,
        progress_bar_class: tqdm | None = None,
        progress_bar_options: dict | None = None,
    ):
        self._dcis, self._start_frames, self._end_frames = [], [], []
        num_frames = 0
        for segment_index in segment_indices:
            dci = SpikeInterfaceRecordingDataChunkIterator(recording=recording, segment_index=segment_index)
            self._dcis.append(dci)
            self._start_frames.append(num_frames)
            num_frames += dci._get_maxshape()[0]
            self._end_frames.append(num_frames)

        super().__init__(
            buffer_gb=buffer_gb,
            buffer_shape=buffer_shape,
            chunk_mb=chunk_mb,
            chunk_shape=chunk_shape,
            display_progress=display_progress,
            progress_bar_class=progress_bar_class,
            progress_bar_options=progress_bar_options,
        )

    def _get_default_chunk_shape(self, chunk_mb: float = 10.0) -> tuple[int, int]:
        return self._dcis[0]._get_default_chunk_shape(chunk_mb=chunk_mb)

    def _get_data(self, selection: tuple[slice]) -> np.ndarray:
        start = selection[0].start
        start = start if start is not None else 0
        stop = selection[0].stop
        stop = stop if stop is not None else self._get_maxshape()[0]
        iterators_range = np.searchsorted(self._end_frames, (start, stop - 1), side="right")
        iterators_spanned = list(range(iterators_range[0], min(iterators_range[-1] + 1, len(self._dcis))))

        # If only one iterator is spanned, we can just return the data from that iterator
        if len(iterators_spanned) == 1:
            relative_start = start - self._start_frames[iterators_spanned[0]]
            relative_stop = stop - self._start_frames[iterators_spanned[0]]
            selection = (slice(relative_start, relative_stop), selection[1])
            return self._dcis[iterators_spanned[0]]._get_data(selection=selection)

        # If multiple iterators are spanned, we need to concatenate the data from each iterator
        channel_start = selection[1].start
        channel_start = channel_start if channel_start is not None else 0
        channel_stop = selection[1].stop
        channel_stop = channel_stop if channel_stop is not None else self._dcis[0]._get_maxshape()[1]
        data = np.empty(shape=(stop - start, channel_stop - channel_start), dtype=self._get_dtype())
        current_frame = 0

        # Left endpoint (first iterator)
        relative_start = start - self._start_frames[iterators_spanned[0]]
        num_frames = self._end_frames[iterators_spanned[0]] - start
        selection = (slice(relative_start, None), selection[1])
        frame_slice = slice(current_frame, current_frame + num_frames)
        data[frame_slice, :] = self._dcis[iterators_spanned[0]]._get_data(selection=selection)
        current_frame += num_frames

        # Inner iterators
        selection = (slice(None, None), selection[1])
        for i in iterators_spanned[1:-1]:
            num_frames = self._end_frames[i] - self._start_frames[i]
            frame_slice = slice(current_frame, current_frame + num_frames)
            data[frame_slice, :] = self._dcis[i]._get_data(selection=selection)
            current_frame += num_frames

        # Right endpoint (last iterator)
        relative_stop = num_frames = stop - self._start_frames[iterators_spanned[-1]]
        selection = (slice(None, relative_stop), selection[1])
        frame_slice = slice(current_frame, current_frame + num_frames)
        data[frame_slice, :] = self._dcis[iterators_spanned[-1]]._get_data(selection=selection)

        return data

    def _get_dtype(self):
        return self._dcis[0]._get_dtype()

    def _get_maxshape(self):
        num_samples = sum([dci._get_maxshape()[0] for dci in self._dcis])
        num_channels = self._dcis[0]._get_maxshape()[1]
        return (num_samples, num_channels)
