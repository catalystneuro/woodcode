"""Reusable single-session conversion logic for the Moore 2025 dataset.

Defines ``session_to_nwb()``, which converts one session of data to NWB, along with the
``get_probe_info_*`` helpers that build probe geometry from the probeinterface library. The driver
scripts (``convert_single_session.py``, ``convert_example_sessions.py``, ``convert_edge_case_sessions.py``)
and ``convert_all_sessions.py`` import ``session_to_nwb`` from here.
"""
from pathlib import Path
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update
import woodcode.nwb as nwb
import probeinterface as pbi
import numpy as np

from moore_2025.temporal_alignment import (
    get_aligned_video_timestamps_juveniles,
    get_aligned_video_timestamps_juveniles_from_dat,
    get_aligned_video_timestamps_adults,
    get_start_time,
)

def get_probe_info_h6b() -> dict:
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

def get_probe_info_h7() -> dict:
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

def get_probe_info_h5() -> dict:
    manufacturer = 'cambridgeneurotech'
    model = 'ASSY-77-H5'
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

def get_probe_info(model: str) -> dict:
    if model == 'Cambridge Neurotech H6b probe':
        return get_probe_info_h6b()
    elif model == 'Cambridge Neurotech H7 probe':
        return get_probe_info_h7()
    elif model == 'Cambridge Neurotech H5 probe':
        return get_probe_info_h5()
    else:
        raise ValueError(f"Unsupported model: {model}")
        # To add more probe types, simply look up the appropriate probe name in the probe interface library.
        # https://github.com/SpikeInterface/probeinterface_library/tree/main


def session_to_nwb(
    *,
    folder_name: str,
    raw_xml_path: Path,
    processed_xml_path: Path,
    nrs_path: Path,
    meta_path: Path,
    mat_path: Path,
    sleep_path: Path,
    lfp_file_path: Path,
    save_path: Path,
    metadata_file_path: Path,
    histology_folder_path: Path,
    timestamps_file_paths: list[Path] | None = None,
    stream_name: str | None = None,
    ttl_stream_name: str | None = None,
    raw_ephys_folder_path: Path | None = None,
    raw_ephys_dat_file_path: Path | None = None,
    video_file_paths: list[Path] | None = None,
    stub_test: bool = False,
    is_adult: bool = True,
):
    """Convert a session to NWB format.

    Parameters
    ----------
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
    timestamps_file_paths : list[Path] | None
        List of paths to timestamp CSV files
    lfp_file_path : Path
        Path to the LFP file
    raw_ephys_folder_path : Path | None
        Path to the raw ephys OpenEphys record node folder
    raw_ephys_dat_file_path : Path | None
        Path to the raw ephys .dat file
    save_path : Path
        Path to save the NWB file
    metadata_file_path : Path
        Path to the metadata YAML file
    video_file_paths : list[Path] | None
        List of paths to video files
    stream_name : str | None, optional
        Name of the Open Ephys stream to read raw ephys data from. Required when
        raw_ephys_folder_path is provided. By default None.
    ttl_stream_name : str | None, optional
        Name of the Open Ephys stream containing the TTL signal for video alignment.
        Required when has_video and has_open_ephys_output. By default None.
    stub_test : bool, optional
        Whether to stub data for testing, by default False
    is_adult : bool, optional
        Whether the subject is an adult or a juvenile, by default True
    """
    has_video = video_file_paths is not None
    has_open_ephys_output = raw_ephys_folder_path is not None
    has_raw_bonsai_output = timestamps_file_paths is not None
    if has_open_ephys_output:
        assert raw_ephys_dat_file_path is None, "Cannot provide both raw_ephys_folder_path and raw_ephys_dat_file_path"
    else:
        assert raw_ephys_dat_file_path is not None, "Must provide either raw_ephys_folder_path or raw_ephys_dat_file_path"
    save_path.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    # load all metadata
    raw_xml_data = nwb.io.read_xml(raw_xml_path)  # load all ephys info from the xml file
    processed_xml_data = nwb.io.read_xml(processed_xml_path)  # load all ephys info from the processed xml file
    lfp_sampling_rate = float(raw_xml_data['eeg_sampling_rate'])
    nrs_data = nwb.io.read_nrs(nrs_path)  # load faulty channel info from the nrs file (i.e. channels not shown in Neuroscope)
    first_video_file_path = video_file_paths[0] if has_video else None
    metadata = nwb.io.read_metadata(meta_path, folder_name, print_output=True)  # Load all metadata from the xlsx file
    first_timestamps_file_path = timestamps_file_paths[0] if timestamps_file_paths else None
    start_time = get_start_time(first_timestamps_file_path, first_video_file_path, folder_name)  # load start time from the first timestamps CSV file

    # Load tracking, epochs and spikes from Matlab files (mostly loaded as pynapple objects)
    matlab_file_path = next(mat_path.glob('TrackingProcessed*.mat'))
    pos = nwb.io.get_matlab_position(matlab_file_path, vbl_name='pos')
    hd = nwb.io.get_matlab_hd(matlab_file_path, vbl_name='ang')
    spikes, waveforms, shank_id = nwb.io.get_matlab_spikes(mat_path)

    # Update metadata with info from metadata.yaml
    metadata_from_yaml = load_dict_from_file(metadata_file_path)
    metadata = dict_deep_update(metadata, metadata_from_yaml)

    # Get probe info
    probe_model = metadata["probe"][0]["type"]
    probe_info = get_probe_info(probe_model)

    # TEMPORAL ALIGNMENT
    # When raw ephys comes from the .dat file, the .dat concatenates segments without preserving
    # the wall-clock gaps between them; the alignment step recovers those gaps from the video LED
    # and produces per-sample timestamps for the .dat on the raw ephys basis.
    raw_ephys_timestamps = None
    if not has_raw_bonsai_output:
        all_aligned_video_timestamps = None
    else:
        if is_adult:
            all_aligned_video_timestamps = get_aligned_video_timestamps_adults(
                timestamp_file_paths=timestamps_file_paths,
                ephys_folder_path=raw_ephys_folder_path,
                ttl_stream_name=ttl_stream_name,
            )
        else:  # juvenile
            if has_open_ephys_output:
                aligned_video_timestamps = get_aligned_video_timestamps_juveniles(
                    timestamp_file_path=timestamps_file_paths[0],
                    ephys_folder_path=raw_ephys_folder_path,
                    ttl_stream_name=ttl_stream_name,
                )
            else:
                aligned_video_timestamps, raw_ephys_timestamps = get_aligned_video_timestamps_juveniles_from_dat(
                    timestamp_file_path=timestamps_file_paths[0],
                    dat_file_path=raw_ephys_dat_file_path,
                    xml_data=raw_xml_data,
                    epoch_ts_file_path=mat_path / "Epoch_TS.csv",
                    ttl_channel_index=metadata["channelTTL"]["0base"],
                )
            all_aligned_video_timestamps = [aligned_video_timestamps]

    # CONSTRUCT NWB FILE
    nwbfile = nwb.convert.create_nwb_file(metadata, start_time)
    nwbfile, raw_xml_data = nwb.convert.add_probes(nwbfile, metadata, raw_xml_data, nrs_data, probe_info)
    if has_open_ephys_output:
        nwbfile = nwb.convert.add_raw_ephys(nwbfile=nwbfile, folder_path=raw_ephys_folder_path, xml_data=raw_xml_data, stream_name=stream_name, stub_test=stub_test)
    else:
        nwbfile = nwb.convert.add_raw_ephys_from_dat(nwbfile=nwbfile, dat_file_path=raw_ephys_dat_file_path, xml_data=raw_xml_data, stub_test=stub_test, timestamps=raw_ephys_timestamps)
    nwbfile = nwb.convert.add_lfp(nwbfile=nwbfile, lfp_path=lfp_file_path, xml_data=raw_xml_data, raw_eseries=nwbfile.acquisition['e-series'], stub_test=stub_test)
    lfp_eseries = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
    nwbfile = nwb.convert.add_tracking(nwbfile, pos, lfp_eseries, lfp_sampling_rate, ang=hd)
    nwbfile, camera_device = nwb.convert.add_camera_device(nwbfile=nwbfile, metadata=metadata)
    if has_video:
        converted_video_file_paths = nwb.video_codec.convert_avi_to_mp4_h264(video_file_paths=video_file_paths, output_directory=save_path)
        # Store the video external_file paths relative to the NWB file (they are written alongside it),
        # as required by NWB best practices / DANDI.
        relative_video_file_paths = [Path(path.name) for path in converted_video_file_paths]
        nwbfile = nwb.convert.add_video(nwbfile=nwbfile, video_file_paths=relative_video_file_paths, all_aligned_video_timestamps=all_aligned_video_timestamps, metadata=metadata, camera_device=camera_device)
    if has_raw_bonsai_output:
        nwbfile = nwb.convert.add_raw_tracking(nwbfile=nwbfile, file_paths=timestamps_file_paths, all_aligned_timestamps=all_aligned_video_timestamps, metadata=metadata, is_adult=is_adult)
    nwbfile = nwb.convert.add_sleep(nwbfile, sleep_path, folder_name, lfp_eseries, lfp_sampling_rate)
    epochs = nwb.convert.get_epochs_from_eseries(eseries=nwbfile.acquisition['e-series'])
    nwbfile = nwb.convert.add_epochs(nwbfile, epochs, metadata)
    nwbfile = nwb.convert.add_units(nwbfile, raw_xml_data, processed_xml_data, spikes, waveforms, shank_id, lfp_eseries, lfp_sampling_rate)  # get shank names from NWB file
    nwbfile = nwb.convert.add_histology(nwbfile, histology_folder_path)

    # save NWB file
    nwb.convert.save_nwb_file(nwbfile, save_path, folder_name)
