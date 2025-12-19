"""Primary script to convert each example session to NWB."""
from pathlib import Path
import pandas as pd
import shutil
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update
import woodcode.nwb as nwb
import probeinterface as pbi
import numpy as np
import re
from datetime import datetime
import pytz
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
from scipy.signal import correlate, correlation_lags

def get_probe_info_juveniles() -> dict:
    # TODO: add logic for daily probe advancement after multi-day example data gets shared. 
    manufacturer = 'cambridgeneurotech'
    model = 'ASSY-37-H6b'
    probe = pbi.get_probe(manufacturer=manufacturer, probe_name=model)

    contact_size = float(probe.contact_shape_params[0]["width"] * probe.contact_shape_params[0]["height"])  # in um^2

    contact_positions = probe.contact_positions

    # Sort by Y coordinate (second column) in descending order for top to bottom
    contact_positions = contact_positions[np.argsort(contact_positions[:, 1])[::-1], :]

    # Flip Y-axis so that down is positive and shift y-axis to start at zero
    contact_positions[:, 1] = -contact_positions[:, 1] + np.max(contact_positions[:, 1])

    # Add empty z-axis (third column)
    contact_positions = np.hstack((contact_positions, np.zeros((contact_positions.shape[0], 1))))

    probe_id = 1
    shank1_id = 1
    probe_info = {
        probe_id: {
            'contact_size': contact_size,
            shank1_id: {
                'electrode_coordinates': contact_positions,
            },
        },
    }
    return probe_info

def get_probe_info_adults() -> dict:
    manufacturer = 'cambridgeneurotech'
    model = 'ASSY-77-H7'
    probe = pbi.get_probe(manufacturer=manufacturer, probe_name=model)

    contact_size = float(probe.contact_shape_params[0]["width"] * probe.contact_shape_params[0]["height"])  # in um^2

    # Split contacts by shank
    shank1_contact_positions = probe.contact_positions[probe.contact_positions[:, 0] < 100, :]
    shank2_contact_positions = probe.contact_positions[probe.contact_positions[:, 0] >= 100, :]

    # Sort by Y coordinate (second column) in descending order for top to bottom
    shank1_contact_positions = shank1_contact_positions[np.argsort(shank1_contact_positions[:, 1])[::-1], :]
    shank2_contact_positions = shank2_contact_positions[np.argsort(shank2_contact_positions[:, 1])[::-1], :]

    # Flip Y-axis so that down is positive and shift y-axis to start at zero
    shank1_contact_positions[:, 1] = -shank1_contact_positions[:, 1] + np.max(shank1_contact_positions[:, 1])
    shank2_contact_positions[:, 1] = -shank2_contact_positions[:, 1] + np.max(shank2_contact_positions[:, 1])

    # Add empty z-axis (third column)
    shank1_contact_positions = np.hstack((shank1_contact_positions, np.zeros((shank1_contact_positions.shape[0], 1))))
    shank2_contact_positions = np.hstack((shank2_contact_positions, np.zeros((shank2_contact_positions.shape[0], 1))))

    probe_id = 1
    shank1_id = 1
    shank2_id = 2
    probe_info = {
        probe_id: {
            'contact_size': contact_size,
            shank1_id: {
                'electrode_coordinates': shank1_contact_positions,
            },
            shank2_id: {
                'electrode_coordinates': shank2_contact_positions,
            },
        },
    }
    return probe_info

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

def get_ttl_timestaps(*, traces: np.ndarray, timestamps: np.ndarray, threshold: float, cooldown_in_seconds: float, sampling_rate: float) -> np.ndarray:
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
    centers = np.concatenate([[centers[0]], centers[1:][center_diff > cooldown_in_samples]])

    ttl_timestamps = timestamps[centers]
    return ttl_timestamps

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
    led_threshold = 2_000
    video_sampling_rate = 30.0
    timestamp_column_name = "Item4.Timestamp"
    led_column_name = "Item3.Val0"
    ttl_threshold = 20_000
    ttl_stream_name = "Rhythm_FPGA-100.0_ADC"
    ttl_channel_id = 'ADC1'
    cooldown_in_seconds = 0.5

    timestamps_df = pd.read_csv(timestamp_file_path, parse_dates=[timestamp_column_name])
    traces = timestamps_df[led_column_name].values
    video_timestamps = np.arange(traces.shape[0]) / video_sampling_rate
    led_timestamps = get_ttl_timestaps(traces=traces, timestamps=video_timestamps, threshold=led_threshold, cooldown_in_seconds=cooldown_in_seconds, sampling_rate=video_sampling_rate)
    led_intervals = np.diff(led_timestamps)

    extractor = OpenEphysBinaryRecordingExtractor(folder_path=ephys_folder_path, stream_name=ttl_stream_name)
    sampling_rate = extractor.get_sampling_frequency()
    ttl_timestamps = np.ones_like(led_timestamps) * np.nan
    for segment_index in range(extractor.get_num_segments()):
        traces = extractor.get_traces(segment_index=segment_index, channel_ids=[ttl_channel_id])
        ephys_timestamps = extractor.get_times(segment_index=segment_index)
        single_segment_ttl_timestamps = get_ttl_timestaps(traces=traces, timestamps=ephys_timestamps, threshold=ttl_threshold, cooldown_in_seconds=cooldown_in_seconds, sampling_rate=sampling_rate)
        ttl_intervals = np.diff(single_segment_ttl_timestamps)
        correlation = correlate(led_intervals, ttl_intervals, mode='full')
        lags = correlation_lags(len(led_intervals), len(ttl_intervals), mode='full')
        best_lag = lags[np.argmax(correlation)]
        assert np.all(np.isnan(ttl_timestamps[best_lag:best_lag + len(single_segment_ttl_timestamps)])), f"Overlap in TTL timestamps at segment {segment_index}"
        ttl_timestamps[best_lag:best_lag + len(single_segment_ttl_timestamps)] = single_segment_ttl_timestamps
        error = led_intervals[best_lag:best_lag + len(ttl_intervals)] - ttl_intervals
        assert np.max(np.abs(error)) < 1.0, f"Alignment error too large: {np.max(np.abs(error))} seconds for segment {segment_index}"

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
        single_segment_ttl_timestamps = get_ttl_timestaps(traces=traces, timestamps=ephys_timestamps, threshold=ttl_threshold, cooldown_in_seconds=cooldown_in_seconds, sampling_rate=sampling_rate)
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
    

def session_to_nwb(
    *,
    dataset_path: Path,
    folder_name: str,
    xml_path: Path,
    nrs_path: Path,
    meta_path: Path,
    mat_path: Path,
    sleep_path: Path,
    video_file_paths: list[Path],
    timestamps_file_paths: list[Path],
    lfp_file_path: Path,
    raw_ephys_folder_path: Path,
    save_path: Path,
    metadata_file_path: Path,
    stub_test: bool = False,
    is_adult: bool = True,
):
    """Convert a session to NWB format.
    
    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory
    folder_name : str
        Name of the session folder
    xml_path : Path
        Path to the XML file
    nrs_path : Path
        Path to the NRS file
    meta_path : Path
        Path to the metadata Excel file
    mat_path : Path
        Path to the Matlab analysis directory
    sleep_path : Path
        Path to the sleep directory
    video_file_paths : list[Path]
        List of paths to video files
    timestamps_file_paths : list[Path]
        List of paths to timestamp CSV files
    lfp_file_path : Path
        Path to the LFP file
    raw_ephys_folder_path : Path
        Path to the raw ephys OpenEphys record node folder
    save_path : Path
        Path to save the NWB file
    metadata_file_path : Path
        Path to the metadata YAML file
    stub_test : bool, optional
        Whether to stub data for testing, by default False
    is_adult : bool, optional
        Whether the subject is an adult or a juvenile, by default True
    """
    if is_adult:
        stream_name = "Rhythm_FPGA-103.0"
        probe_info = get_probe_info_adults()
        all_aligned_video_timestamps = get_aligned_video_timestamps_adults(
            timestamp_file_paths=timestamps_file_paths,
            ephys_folder_path=raw_ephys_folder_path,
        )
    else: # juvenile
        stream_name = "Rhythm_FPGA-100.0"
        probe_info = get_probe_info_juveniles()
        aligned_video_timestamps = get_aligned_video_timestamps_juveniles(
            timestamp_file_path=timestamps_file_paths[0],
            ephys_folder_path=raw_ephys_folder_path,
        )
        all_aligned_video_timestamps = [aligned_video_timestamps]

    save_path.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    # load all metadata
    xml_data = nwb.io.read_xml(xml_path)  # load all ephys info from the xml file
    lfp_sampling_rate = float(xml_data['eeg_sampling_rate'])
    nrs_data = nwb.io.read_nrs(nrs_path)  # load faulty channel info from the nrs file (i.e. channels not shown in Neuroscope)
    metadata = nwb.io.read_metadata(meta_path, folder_name, print_output=True)  # Load all metadata from the xlsx file
    start_time = get_start_time(timestamps_file_paths[0])  # load start time from the first timestamps CSV file

    # Load tracking, epochs and spikes from Matlab files (mostly loaded as pynapple objects)
    pos = nwb.io.get_matlab_position(mat_path / 'TrackingProcessed_Final.mat', vbl_name='pos')
    hd = nwb.io.get_matlab_hd(mat_path / 'TrackingProcessed_Final.mat', vbl_name='ang')
    spikes, waveforms, shank_id = nwb.io.get_matlab_spikes(mat_path)

    # Update metadata with info from metadata.yaml
    metadata_from_yaml = load_dict_from_file(metadata_file_path)
    metadata = dict_deep_update(metadata, metadata_from_yaml)

    # CONSTRUCT NWB FILE
    nwbfile = nwb.convert.create_nwb_file(metadata, start_time)
    nwbfile = nwb.convert.add_probes(nwbfile, metadata, xml_data, nrs_data, probe_info)
    nwbfile = nwb.convert.add_tracking(nwbfile, pos, hd)
    nwbfile = nwb.convert.add_raw_tracking(nwbfile=nwbfile, file_paths=timestamps_file_paths, all_aligned_timestamps=all_aligned_video_timestamps, is_adult=is_adult)
    nwbfile = nwb.convert.add_video(nwbfile=nwbfile, video_file_paths=video_file_paths, all_aligned_video_timestamps=all_aligned_video_timestamps, metadata=metadata)
    nwbfile = nwb.convert.add_raw_ephys(nwbfile=nwbfile, folder_path=raw_ephys_folder_path, xml_data=xml_data, stream_name=stream_name, stub_test=stub_test)
    nwbfile = nwb.convert.add_lfp(nwbfile=nwbfile, lfp_path=lfp_file_path, xml_data=xml_data, raw_eseries=nwbfile.acquisition['e-series'], stub_test=stub_test)
    lfp_eseries = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
    nwbfile = nwb.convert.add_sleep(nwbfile, sleep_path, folder_name, lfp_eseries, lfp_sampling_rate)
    epochs = nwb.convert.get_epochs_from_eseries(eseries=nwbfile.acquisition['e-series'])
    nwbfile = nwb.convert.add_epochs(nwbfile, epochs, metadata)
    nwbfile = nwb.convert.add_units(nwbfile, xml_data, spikes, waveforms, shank_id, lfp_eseries, lfp_sampling_rate)  # get shank names from NWB file

    # save NWB file
    nwb.convert.save_nwb_file(nwbfile, save_path, folder_name)


def main():
    """Define paths and convert example sessions to NWB."""
    stub_test = False
    dataset_path = Path('/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset')
    output_folder_path = Path('/Volumes/T7/CatalystNeuro/Spyglass/raw')
    if output_folder_path.exists():
        shutil.rmtree(output_folder_path)

    # Example Juvenile Sessions
    juvenile_folder_path = dataset_path / "H3000_Juveniles"
    metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/juvenile_metadata.yaml")

    # Example Juvenile WT session
    jv_wt_folder_path = juvenile_folder_path / "WT"
    folder_name = 'H3022-210805'
    xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    meta_path = dataset_path / 'MooreDataset_Metadata.xlsx'  # path to metadata file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-08-05T17_06_24.avi",
    ]
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "Bonsai testing2021-08-05T17_06_23.csv",
    ]
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_wt_folder_path / folder_name / "Raw"
    save_path = output_folder_path

    session_to_nwb(
        dataset_path=dataset_path,
        folder_name=folder_name,
        xml_path=xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=metadata_file_path,
        stub_test=stub_test,
        is_adult=False,
    )

    # Example Juvenile KO session
    jv_ko_folder_path = juvenile_folder_path / "KO"
    folder_name = 'H3016-210423'
    # Note: H3016-210423 uses H3022-210805's XML because the original had faulty channels removed from spikeDetection (26 vs 32 channels). Both sessions share the same probe mapping.
    xml_path = jv_wt_folder_path / 'H3022-210805' / "Processed" / ( 'H3022-210805' + '.xml')
    nrs_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    meta_path = dataset_path / 'MooreDataset_Metadata.xlsx'  # path to metadata file
    mat_path = jv_ko_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_ko_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-04-23T14_14_05.avi",
    ]
    timestamps_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "Bonsai testing2021-04-23T14_13_55.csv",
    ]
    lfp_file_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_ko_folder_path / folder_name / "Raw"
    save_path = output_folder_path

    session_to_nwb(
        dataset_path=dataset_path,
        folder_name=folder_name,
        xml_path=xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=metadata_file_path,
        stub_test=stub_test,
        is_adult=False,
    )

    # Example Adult Sessions
    adult_folder_path = dataset_path / "H4800_Adults"
    metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/adult_metadata.yaml")

    # Example Adult WT session
    adult_wt_folder_path = adult_folder_path / "WT"
    folder_name = 'H4813-220728'
    # Note: using the XML from the Raw folder here since the one in Processed is missing one of the channels for shank 2
    xml_path = adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-103.0" / "continuous.xml"
    nrs_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    meta_path = dataset_path / 'MooreDataset_Metadata.xlsx'  # path to metadata file
    mat_path = adult_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = adult_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiVideo2022-07-28T18_14_29.avi",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiVideo2022-07-28T18_37_00.avi",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiVideo2022-07-28T20_09_04.avi",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiVideo2022-07-28T20_33_32.avi",
    ]
    timestamps_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiTracking2022-07-28T18_14_27.csv",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiTracking2022-07-28T18_36_58.csv",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiTracking2022-07-28T20_08_59.csv",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiTracking2022-07-28T20_33_30.csv",
    ]
    lfp_file_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = adult_wt_folder_path / folder_name / "Raw"
    save_path = output_folder_path
    session_to_nwb(
        dataset_path=dataset_path,
        folder_name=folder_name,
        xml_path=xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=metadata_file_path,
        stub_test=stub_test,
        is_adult=True,
    )

    # Example Adult KO session
    adult_ko_folder_path = adult_folder_path / "KO"
    folder_name = 'H4817-220828'
    xml_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    meta_path = dataset_path / 'MooreDataset_Metadata.xlsx'  # path to metadata file
    mat_path = adult_ko_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = adult_ko_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiVideo2022-08-28T16_18_06.avi",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiVideo2022-08-28T16_40_58.avi",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiVideo2022-08-28T18_13_23.avi",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiVideo2022-08-28T18_36_14.avi",
    ]
    timestamps_file_paths = [
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiTracking2022-08-28T16_18_05.csv",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiTracking2022-08-28T16_40_56.csv",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiTracking2022-08-28T18_13_20.csv",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiTracking2022-08-28T18_36_12.csv",
    ]
    lfp_file_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = adult_ko_folder_path / folder_name / "Raw"
    save_path = output_folder_path
    session_to_nwb(
        dataset_path=dataset_path,
        folder_name=folder_name,
        xml_path=xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=metadata_file_path,
        stub_test=stub_test,
        is_adult=True,
    )


if __name__ == "__main__":
    main()
