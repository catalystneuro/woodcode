"""Temporal alignment for the Duszkiewicz 2026 dataset.

Each session is split across two OpenEphys experiments (epochs) that were recorded by fully
stopping and restarting acquisition, so each experiment has its own clock starting at 0. Video
frames are synchronized to ephys per experiment by anchoring the Bonsai camera clock to the
ephys clock with the first camera TTL pulse, then placed on the unified session time basis using
each experiment's sync offset (recovered from ``sync_messages.txt`` Software Time by the
orchestrator).

Bonsai records substantially more camera TTL pulses than there are saved video frames (~5-8%
surplus) because of frames missed by Bonsai, and the surplus pulses cannot be matched to specific
frames. Per the Dudchenko lab (Q35), we therefore do not attempt per-pulse-to-per-frame matching.
Instead we use the timestamp of the first TTL pulse in each epoch to offset the Bonsai timestamps
onto the ephys clock: the Bonsai timestamps are themselves evenly spaced at the true frame times,
so converting them to seconds relative to the first frame and adding the first pulse's ephys time
places the whole video stream on the ephys clock.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor

import woodcode.nwb as nwb
from moore_2025.temporal_alignment import get_ttl_timestamps


def get_aligned_video_timestamps_duszkiewicz(
    *,
    timestamp_file_paths: list[Path],
    record_node_path: Path,
    ttl_stream_name: str,
    experiment_names: list[str],
    sync_offsets: list[float],
    ttl_channel_id: str = "ADC1",
    ttl_threshold: float = 10_000,
) -> list[np.ndarray]:
    """Align video timestamps to the unified ephys time basis, one experiment at a time.

    The Bonsai camera timestamps are anchored to the ephys clock using the first camera TTL pulse
    of each experiment, rather than by matching individual pulses to individual frames (see module
    docstring / Q35).

    Parameters
    ----------
    timestamp_file_paths : list[Path]
        Bonsai tracking CSV files, one per experiment (in experiment order).
    record_node_path : Path
        Path to the OpenEphys ``Record Node`` folder containing the experiments.
    ttl_stream_name : str
        Name of the OpenEphys ADC stream carrying the LED sync pulses.
    experiment_names : list[str]
        OpenEphys experiment folder names (e.g. ``["experiment1", "experiment2"]``).
    sync_offsets : list[float]
        Per-experiment offset (seconds) onto the unified session basis, from the Software Times.
    ttl_channel_id : str, optional
        ADC channel carrying the LED pulses, by default ``"ADC1"``.
    ttl_threshold : float, optional
        Threshold (raw ADC units) for detecting LED pulse onsets, by default 10000.

    Returns
    -------
    list[np.ndarray]
        Aligned video timestamps for each experiment.
    """
    print("Aligning Duszkiewicz video timestamps...")
    timestamp_column_name = "Item3.Timestamp"
    cooldown_in_seconds = 0.0

    all_aligned_video_timestamps = []
    for experiment_name, timestamp_file_path, sync_offset in zip(
        experiment_names, timestamp_file_paths, sync_offsets, strict=True
    ):
        print(f"  Aligning {experiment_name} with timestamp file {timestamp_file_path.name}...")
        extractor = OpenEphysBinaryRecordingExtractor(
            folder_path=record_node_path, stream_name=ttl_stream_name, experiment_name=experiment_name
        )
        sampling_rate = extractor.get_sampling_frequency()
        separator = nwb.convert.get_separator(file_path=timestamp_file_path)
        timestamps_df = pd.read_csv(timestamp_file_path, parse_dates=[timestamp_column_name], sep=separator)

        # Detect the camera TTL pulses on the unified ephys basis; only the first pulse is used to
        # anchor the Bonsai clock (see module docstring / Q35).
        traces = extractor.get_traces(segment_index=0, channel_ids=[ttl_channel_id])
        ephys_timestamps = extractor.get_times(segment_index=0) + sync_offset
        ttl_timestamps = get_ttl_timestamps(
            traces=traces,
            timestamps=ephys_timestamps,
            threshold=ttl_threshold,
            cooldown_in_seconds=cooldown_in_seconds,
            sampling_rate=sampling_rate,
        )
        first_ttl_timestamp = ttl_timestamps[0]

        # Offset the Bonsai timestamps onto the ephys clock: seconds relative to the first frame,
        # plus the ephys time of the first TTL pulse (which corresponds to that first frame).
        bonsai_timestamps = timestamps_df[timestamp_column_name]
        relative_seconds = (bonsai_timestamps - bonsai_timestamps.iloc[0]).dt.total_seconds().to_numpy()
        aligned_video_timestamps = relative_seconds + first_ttl_timestamp

        # The video has one extra frame at the end relative to the Bonsai tracking rows, so
        # extrapolate one more frame time (one median inter-frame interval) to match the video frame
        # count. add_raw_tracking drops this trailing timestamp when aligning the tracking data.
        median_interval = np.median(np.diff(aligned_video_timestamps))
        aligned_video_timestamps = np.append(
            aligned_video_timestamps, aligned_video_timestamps[-1] + median_interval
        )

        all_aligned_video_timestamps.append(aligned_video_timestamps)

    return all_aligned_video_timestamps
