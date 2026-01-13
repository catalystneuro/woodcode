from pathlib import Path
import pandas as pd
import numpy as np
import re
from datetime import datetime
import pytz
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
from time import time

def get_start_time(timestamps_file_path: Path) -> str:
    """
    Get the session start datetime from the name of the timestamps CSV file.
    
    Parameters
    ----------
    timestamps_file_path : Path
        Path to the timestamps CSV file.
    
    Returns
    -------
    datetime
        The session start datetime with timezone information.
    """
    # Example filename: 'Bonsai testing2021-08-05T17_06_23.csv'
    filename = timestamps_file_path.stem  # Ex. 'Bonsai testing2021-08-05T17_06_23'
    pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Could not extract datetime from filename: {filename}")
    
    start_time = match.group(1)
    start_time = datetime.strptime(start_time, '%Y-%m-%dT%H_%M_%S')
    tz_info = pytz.timezone('Europe/London')
    start_time = tz_info.localize(start_time)

    return start_time

def get_ttl_timestamps(*, traces: np.ndarray, timestamps: np.ndarray, threshold: float, cooldown_in_seconds: float, sampling_rate: float) -> np.ndarray:
    """
    Get TTL timestamps from traces.

    Parameters
    ----------
    traces : np.ndarray
        The input signal traces.
    timestamps : np.ndarray
        The corresponding timestamps for the traces.
    threshold : float
        The threshold to detect TTL onsets and offsets.
    cooldown_in_seconds : float
        The cooldown period in seconds to avoid detecting the same onset multiple times.
    sampling_rate : float
        The sampling rate of the traces in Hz.
    
    Returns
    -------
    np.ndarray
        The timestamps of detected TTL centers.
    """    

    peak_points = np.where((traces > threshold))[0]
    gaps = np.diff(peak_points) > 1
    onsets = np.concatenate([[peak_points[0]], peak_points[1:][gaps]])
    offsets = np.concatenate([peak_points[:-1][gaps], [peak_points[-1]]])
    onsets = np.array(onsets)
    offsets = np.array(offsets)
    centers = (onsets + offsets) // 2

    # Add a cooldown period to avoid detecting the same onset multiple times
    cooldown_in_samples = int(cooldown_in_seconds * sampling_rate)
    center_diff = np.diff(centers)
    cooldown_conflict_mask = center_diff <= cooldown_in_samples
    cooldown_conflict_mask[:-1] = cooldown_conflict_mask[:-1] | np.roll(cooldown_conflict_mask[:-1], -1) # Exclude all centers involved in a conflict even if they are the first in the group
    cooldown_conflict_mask = np.concatenate(([False], cooldown_conflict_mask))
    if center_diff[0] <= cooldown_in_samples:
        cooldown_conflict_mask[0] = True
    centers = centers[~cooldown_conflict_mask]

    ttl_timestamps = timestamps[centers]
    return ttl_timestamps

def find_putative_interval_match(*, led_intervals: np.ndarray, ttl_intervals: np.ndarray, tolerance_in_seconds: float) -> tuple[int, int]:
    """
    Find the first matching interval between LED intervals and TTL intervals within a specified tolerance.

    Parameters
    ----------
    led_intervals: np.ndarray
        Array of LED event intervals.
    ttl_intervals: np.ndarray
        Array of TTL event intervals.
    tolerance_in_seconds: float
        Tolerance in seconds for matching intervals.

    Returns
    -------
    tuple[int, int]
        A tuple containing the indices of the matching LED interval and TTL interval.
        If no match is found, returns (None, None).
    """
    potential_first_ttl_intervals = [sum(ttl_intervals[:i+1]) for i in range(len(ttl_intervals))]
    for led_index, led_interval in enumerate(led_intervals):
        for ttl_index, ttl_interval in enumerate(potential_first_ttl_intervals):
            if led_interval - ttl_interval > tolerance_in_seconds:
                continue
            elif led_interval - ttl_interval >= -1 * tolerance_in_seconds and led_interval - ttl_interval <= tolerance_in_seconds:
                return led_index, ttl_index
            elif led_interval - ttl_interval < -1 * tolerance_in_seconds:
                break
    return None, None

def check_interval_match(*, match_led_index, match_ttl_index, led_intervals, ttl_intervals, min_matches, tolerance_in_seconds):
    """
    Check if a sequence of intervals match between LED and TTL intervals starting from given indices.

    Parameters
    ----------
    match_led_index: int
        The starting index in LED intervals.
    match_ttl_index: int
        The starting index in TTL intervals.
    led_intervals: np.ndarray
        Array of LED event intervals.
    ttl_intervals: np.ndarray
        Array of TTL event intervals.
    min_matches: int
        Minimum number of matching intervals required to consider an alignment valid.
    tolerance_in_seconds: float
        Tolerance in seconds for matching intervals.

    Returns
    -------
    bool
        True if the required number of intervals match, False otherwise.
    """
    global_ttl_index = match_ttl_index
    for next_led_index in range(match_led_index+1, match_led_index + min_matches):
        next_ttl_index = global_ttl_index + 1
        next_led_interval = led_intervals[next_led_index]
        _, ttl_index = find_putative_interval_match(led_intervals=[next_led_interval], ttl_intervals=ttl_intervals[next_ttl_index:], tolerance_in_seconds=tolerance_in_seconds)
        if ttl_index is None:
            return False
        global_ttl_index = ttl_index + next_ttl_index
    return True


def find_segment_start(*, led_times: np.ndarray, ttl_times: np.ndarray, min_matches: int, tolerance_in_seconds: float) -> int:
    """
    Find the index of the LED times that best aligns with the start of the TTL times for this segment.

    Parameters
    ----------

    led_times: np.ndarray
        Array of LED event timestamps.
    ttl_times: np.ndarray
        Array of TTL event timestamps.
    min_matches: int
        Minimum number of matching intervals required to consider an alignment valid.
    tolerance_in_seconds: float
        Tolerance in seconds for matching intervals.

    Returns
    -------
    int
        The starting index in led_times that best aligns with ttl_times.
    """
    led_intervals = np.diff(led_times)
    ttl_intervals = np.diff(ttl_times)

    match_is_found = False
    global_led_index = 0
    while not match_is_found:
        led_index, ttl_index = find_putative_interval_match(led_intervals=led_intervals, ttl_intervals=ttl_intervals, tolerance_in_seconds=tolerance_in_seconds)
        if led_index is None:
            raise ValueError("No matching intervals found between LED and TTL times.")
        global_led_index += led_index
        match_is_found = check_interval_match(
            match_led_index=led_index,
            match_ttl_index=ttl_index,
            led_intervals=led_intervals,
            ttl_intervals=ttl_intervals,
            min_matches=min_matches,
            tolerance_in_seconds=tolerance_in_seconds,
        )
        if match_is_found:
            return global_led_index
        
        led_intervals = led_intervals[led_index + 1:]
        global_led_index += 1

def correct_ttl_times(*, led_times: np.ndarray, ttl_times: np.ndarray, min_matches: int, tolerance_in_seconds: float) -> np.ndarray:
    """
    Correct TTL times to align with LED times starting from a given index.

    Parameters
    ----------
    led_times: np.ndarray
        Array of LED event timestamps.
    ttl_times: np.ndarray
        Array of TTL event timestamps.
    min_matches: int
        Minimum number of matching intervals required to consider an alignment valid.
    tolerance_in_seconds: float
        Tolerance in seconds for matching intervals.

    Returns
    -------
    np.ndarray
        The corrected TTL timestamps.
    """
    ttl_intervals = np.diff(ttl_times)
    corrected_ttl_timestamps = [] # TTL timestamps with some number of dropped pulses to account for dropped LED flashes.
    segment_start_index = find_segment_start(led_times=led_times, ttl_times=ttl_times, min_matches=min_matches, tolerance_in_seconds=tolerance_in_seconds)
    segment_led_times = led_times[segment_start_index:]
    segment_led_intervals = np.diff(segment_led_times)

    corrected_ttl_timestamps.append(ttl_times[0])
    previous_ttl_index = 0
    previous_ttl_timestamp = ttl_times[0]
    num_intervals = min(len(segment_led_intervals), len(ttl_intervals))
    for led_interval_index in range(num_intervals):
        led_interval = segment_led_intervals[led_interval_index]
        for ttl_timestamp_index in range(previous_ttl_index+1, len(ttl_times)):
            next_ttl_timestamp = ttl_times[ttl_timestamp_index]
            ttl_interval = next_ttl_timestamp - previous_ttl_timestamp
            if led_interval - ttl_interval > tolerance_in_seconds:
                continue
            elif led_interval - ttl_interval < tolerance_in_seconds or led_interval - ttl_interval > -1 * tolerance_in_seconds:
                corrected_ttl_timestamps.append(next_ttl_timestamp)
                previous_ttl_timestamp = next_ttl_timestamp
                previous_ttl_index = ttl_timestamp_index
                break
            else:
                raise ValueError(f"No matching TTL interval found for LED interval {led_interval} at index {led_interval_index}")
    corrected_ttl_timestamps = np.array(corrected_ttl_timestamps)

    # If the last interval does not match, there is no way to correct it with a compound interval, so we drop it.
    last_ttl_interval = corrected_ttl_timestamps[-1] - corrected_ttl_timestamps[-2]
    last_led_interval = segment_led_times[len(corrected_ttl_timestamps)-1] - segment_led_times[len(corrected_ttl_timestamps)-2]
    if np.abs(last_ttl_interval - last_led_interval) > tolerance_in_seconds:
        corrected_ttl_timestamps = corrected_ttl_timestamps[:-1]

    return corrected_ttl_timestamps, segment_start_index

def align_by_interpolation(unaligned_dense_timestamps: np.ndarray, unaligned_sparse_timestamps: np.ndarray, aligned_sparse_timestamps: np.ndarray) -> np.ndarray:
    """
    Interpolate timestamps using a mapping from an unaligned time basis to an aligned one.

    Use this function when the unaligned timestamps of data are not directly tracked by a primary
    system, but are known to occur between timestamps that are tracked. The timestamps are aligned
    by interpolating between the two time bases.

    An example could be a metronomic TTL pulse (e.g., every second) from a secondary data stream to the primary
    timing system; if the time references are recorded within the relative time of the secondary
    data stream, then their exact time in the primary system is inferred given the pulse times.

    Parameters
    ----------
    unaligned_dense_timestamps : np.ndarray
        The dense timestamps of the unaligned secondary time basis that need to be aligned.
    unaligned_sparse_timestamps : np.ndarray
        The timestamps of the unaligned secondary time basis.
    aligned_sparse_timestamps : np.ndarray
        The timestamps aligned to the primary time basis.

    Returns
    -------
    np.ndarray
        The aligned dense timestamps.
    """
    aligned_dense_timestamps = np.interp(
        x=unaligned_dense_timestamps,
        xp=unaligned_sparse_timestamps,
        fp=aligned_sparse_timestamps
    )
    return aligned_dense_timestamps

def get_aligned_video_timestamps_juveniles(
    *,
    timestamp_file_path: Path,
    ephys_folder_path: Path,
) -> np.ndarray:
    """
    Get aligned video timestamps for juvenile sessions.

    Parameters
    ----------
    timestamp_file_path : Path
        Path to the video timestamps CSV file.
    ephys_folder_path : Path
        Path to the ephys OpenEphys record node folder.

    Returns
    -------
    np.ndarray
        The aligned video timestamps.
    """
    print("Aligning juvenile video timestamps...")
    led_threshold = 2_100
    video_sampling_rate = 30.0
    timestamp_column_name = "Item4.Timestamp"
    led_column_name = "Item3.Val0"
    ttl_threshold = 20_000
    ttl_stream_name = "Rhythm_FPGA-100.0_ADC"
    ttl_channel_id = 'ADC1'
    cooldown_in_seconds = 1.0
    min_matches = 5
    tolerance_in_seconds = 0.5

    timestamps_df = pd.read_csv(timestamp_file_path, parse_dates=[timestamp_column_name])
    traces = timestamps_df[led_column_name].values
    video_timestamps = np.arange(traces.shape[0]) / video_sampling_rate
    led_timestamps = get_ttl_timestamps(traces=traces, timestamps=video_timestamps, threshold=led_threshold, cooldown_in_seconds=cooldown_in_seconds, sampling_rate=video_sampling_rate)
    led_intervals = np.diff(led_timestamps)

    extractor = OpenEphysBinaryRecordingExtractor(folder_path=ephys_folder_path, stream_name=ttl_stream_name)
    sampling_rate = extractor.get_sampling_frequency()
    ttl_timestamps = np.ones_like(led_timestamps) * np.nan
    for segment_index in range(extractor.get_num_segments()):
        print(f"  Aligning segment {segment_index}...")
        t0 = time()
        traces = extractor.get_traces(segment_index=segment_index, channel_ids=[ttl_channel_id])
        ephys_timestamps = extractor.get_times(segment_index=segment_index)
        single_segment_ttl_timestamps = get_ttl_timestamps(traces=traces, timestamps=ephys_timestamps, threshold=ttl_threshold, cooldown_in_seconds=cooldown_in_seconds, sampling_rate=sampling_rate)
        single_segment_ttl_timestamps, segment_start_index = correct_ttl_times(led_times=led_timestamps, ttl_times=single_segment_ttl_timestamps, min_matches=min_matches, tolerance_in_seconds=tolerance_in_seconds)
        
        ttl_intervals = np.diff(single_segment_ttl_timestamps)
        assert np.all(np.isnan(ttl_timestamps[segment_start_index:segment_start_index + len(single_segment_ttl_timestamps)])), f"Overlap in TTL timestamps at segment {segment_index}"
        ttl_timestamps[segment_start_index:segment_start_index + len(single_segment_ttl_timestamps)] = single_segment_ttl_timestamps
        error = led_intervals[segment_start_index:segment_start_index + len(ttl_intervals)] - ttl_intervals
        assert np.max(np.abs(error)) < tolerance_in_seconds, f"Alignment error too large: {np.max(np.abs(error))} seconds for segment {segment_index}"
        print(f"    Segment {segment_index} aligned in {time() - t0:.2f} seconds")

    # NaN values represent LED pulses that were not recorded in the ephys data (ex. between segments)
    not_nan = ~np.isnan(ttl_timestamps)
    led_timestamps = led_timestamps[not_nan]
    ttl_timestamps = ttl_timestamps[not_nan]

    aligned_video_timestamps = align_by_interpolation(
        unaligned_dense_timestamps=video_timestamps,
        unaligned_sparse_timestamps=led_timestamps,
        aligned_sparse_timestamps=ttl_timestamps,
    )

    return aligned_video_timestamps


def get_aligned_video_timestamps_adults(
    *,
    timestamp_file_paths: list[Path],
    ephys_folder_path: Path,
) -> list[np.ndarray]:
    """
    Get aligned video timestamps for adult sessions.

    Parameters
    ----------
    timestamp_file_paths : list[Path]
        List of paths to the video timestamps CSV files.
    ephys_folder_path : Path
        Path to the ephys OpenEphys record node folder.

    Returns
    -------
    list[np.ndarray]
        The aligned video timestamps for each segment/file.
    """
    timestamp_column_name = "Item3.Timestamp"
    ttl_threshold = 10_000
    ttl_stream_name = "Rhythm_FPGA-103.0_ADC"
    ttl_channel_id = 'ADC6'
    cooldown_in_seconds = 0.0

    extractor = OpenEphysBinaryRecordingExtractor(folder_path=ephys_folder_path, stream_name=ttl_stream_name)
    sampling_rate = extractor.get_sampling_frequency()
    assert extractor.get_num_segments() == len(timestamp_file_paths), f"Number of ephys segments ({extractor.get_num_segments()}) does not match number of timestamp files ({len(timestamp_file_paths)})."
    all_aligned_video_timestamps = []

    for segment_index, timestamp_file_path in enumerate(timestamp_file_paths):
        timestamps_df = pd.read_csv(timestamp_file_path, parse_dates=[timestamp_column_name])
        num_frames = timestamps_df.shape[0] + 1 # + 1 bc video has one extra frame at the end
        traces = extractor.get_traces(segment_index=segment_index, channel_ids=[ttl_channel_id])
        ephys_timestamps = extractor.get_times(segment_index=segment_index)
        single_segment_ttl_timestamps = get_ttl_timestamps(traces=traces, timestamps=ephys_timestamps, threshold=ttl_threshold, cooldown_in_seconds=cooldown_in_seconds, sampling_rate=sampling_rate)
        num_ttls = single_segment_ttl_timestamps.shape[0]
        assert num_ttls >= num_frames, f"Number of TTLs ({num_ttls}) is less than number of video frames ({num_frames}) for segment {segment_index}."

        # Correct for dropped frames by removing the TTLs that correspond to the largest gaps
        num_dropped_frames = num_ttls - num_frames
        timestamps_df["seconds"] = (timestamps_df[timestamp_column_name] - timestamps_df[timestamp_column_name].iloc[0]).dt.total_seconds()
        t_diff = np.diff(timestamps_df["seconds"])
        drop_indices = np.argsort(t_diff)[-num_dropped_frames:]  # indices of largest gaps
        drop_indices = np.sort(drop_indices)  # sort chronologically
        ttl_indices_to_remove = drop_indices + 1 # since drop happens AFTER that CSV frame
        single_segment_ttl_timestamps = np.delete(single_segment_ttl_timestamps, ttl_indices_to_remove)

        all_aligned_video_timestamps.append(single_segment_ttl_timestamps)

    return all_aligned_video_timestamps