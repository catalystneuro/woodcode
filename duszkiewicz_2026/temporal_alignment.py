"""Temporal alignment for the Duszkiewicz 2026 dataset.

Each session is split across two OpenEphys experiments (epochs) that were recorded by fully
stopping and restarting acquisition, so each experiment has its own clock starting at 0. Video
frames are synchronized to ephys per experiment via the LED TTL pulses recorded on an ADC
channel, then placed on the unified session time basis using each experiment's sync offset
(recovered from ``sync_messages.txt`` Software Time by the orchestrator).
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
        num_frames = timestamps_df.shape[0] + 1  # + 1 bc the video has one extra frame at the end

        traces = extractor.get_traces(segment_index=0, channel_ids=[ttl_channel_id])
        ephys_timestamps = extractor.get_times(segment_index=0)
        ephys_timestamps = ephys_timestamps + sync_offset
        single_segment_ttl_timestamps = get_ttl_timestamps(
            traces=traces,
            timestamps=ephys_timestamps,
            threshold=ttl_threshold,
            cooldown_in_seconds=cooldown_in_seconds,
            sampling_rate=sampling_rate,
        )
        num_ttls = single_segment_ttl_timestamps.shape[0]
        assert num_ttls >= num_frames, (
            f"Number of TTLs ({num_ttls}) is less than number of video frames ({num_frames}) "
            f"for {experiment_name}."
        )

        # The camera emits ~5-8% more TTL pulses than there are saved frames and the surplus can't
        # be matched to specific frames, so we take the first num_frames pulses (assumes the extra
        # pulses are at the end of the epoch). Pending confirmation from the Dudchenko lab (Q35).
        # TODO: This first-num_frames truncation is a temporary stopgap. Replace it with the correct
        # frame-to-TTL matching once Adrian confirms how the surplus pulses map to dropped/extra
        # frames (e.g. whether the extras are genuinely at the end or interspersed).
        single_segment_ttl_timestamps = single_segment_ttl_timestamps[:num_frames]

        all_aligned_video_timestamps.append(single_segment_ttl_timestamps)

    return all_aligned_video_timestamps
