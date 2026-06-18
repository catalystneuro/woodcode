"""Single-session conversion logic for the Duszkiewicz 2026 dataset.

Defines ``session_to_nwb()`` for one Duszkiewicz session and ``get_probe_info_duszkiewicz()``,
which builds the two-probe geometry. The session is organized like the Moore H4800 adults, with
these differences handled here:

- Two Cambridge Neurotech H7 probes (4 shanks / 128 ch) in different brain regions.
- Two epochs recorded as separate OpenEphys experiments, each with its own clock starting at 0;
  they are joined on a unified time basis using the Software-Time offset from ``sync_messages.txt``.
- Raw digital TTL events (epoch 2) stored as Spyglass-readable DIO TimeSeries.
- Processed cue events (``epCue1``-``epCue4``, stored as DIO TimeSeries) and blink timestamps from
  ``CueEpochs.mat``.
"""
from datetime import datetime
from pathlib import Path

import pytz
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update

import woodcode.nwb as nwb
from moore_2025.session_to_nwb import get_probe_info_h7
from duszkiewicz_2026.temporal_alignment import get_aligned_video_timestamps_duszkiewicz


def get_probe_info_duszkiewicz(metadata: dict) -> dict:
    """Build probe geometry for the two-H7-probe Duszkiewicz layout.

    Both probes are Cambridge Neurotech H7 (2 shanks each). Reuses the H7 per-shank coordinate
    computation from the Moore module and emits the nested structure ``add_probes`` expects, with
    global shank ids running 1..4 across the two probes
    (``{1: {contact_size, 1, 2}, 2: {contact_size, 3, 4}}``).
    """
    h7_probe_info = get_probe_info_h7()  # {1: {'contact_size', 1: {...}, 2: {...}}}
    contact_size = h7_probe_info[1]["contact_size"]
    shank_coordinates = {
        1: h7_probe_info[1][1]["electrode_coordinates"],
        2: h7_probe_info[1][2]["electrode_coordinates"],
    }

    probe_info = {}
    global_shank_id = 1
    for probe_metadata in metadata["probe"]:
        assert probe_metadata["type"] == "Cambridge Neurotech H7 probe", (
            f"get_probe_info_duszkiewicz only supports H7 probes, got {probe_metadata['type']}"
        )
        probe_id = probe_metadata["id"]
        probe_info[probe_id] = {"contact_size": contact_size}
        for local_shank_id in range(1, probe_metadata["nshanks"] + 1):
            probe_info[probe_id][global_shank_id] = {
                "electrode_coordinates": shank_coordinates[local_shank_id],
            }
            global_shank_id += 1
    return probe_info


def session_to_nwb(
    *,
    folder_name: str,
    metadata_lookup_name: str,
    record_node_path: Path,
    experiment_names: list[str],
    stream_name: str,
    ttl_stream_name: str,
    raw_xml_path: Path,
    processed_xml_path: Path,
    nrs_path: Path,
    lfp_file_path: Path,
    meta_path: Path,
    metadata_file_path: Path,
    mat_path: Path,
    sleep_path: Path,
    cue_epochs_path: Path,
    ttl_folder_path: Path,
    video_file_paths: list[Path],
    timestamps_file_paths: list[Path],
    save_path: Path,
    stub_test: bool = False,
):
    """Convert one Duszkiewicz session to NWB.

    Parameters
    ----------
    folder_name : str
        Session folder name / NWB identifier (e.g. ``"H6813-240605"``, dash-separated).
    metadata_lookup_name : str
        Key used to look up the row in the metadata xlsx (e.g. ``"H6813_240605"``, underscore).
    record_node_path : Path
        Path to the OpenEphys ``Record Node`` folder containing the two experiments.
    experiment_names : list[str]
        Experiment folder names in epoch order (e.g. ``["experiment1", "experiment2"]``).
    stream_name : str
        Name of the neural OpenEphys stream (prefixed with the record node, e.g.
        ``"Record Node 101#Acquisition_Board-100.Rhythm Data"``).
    ttl_stream_name : str
        Name of the ADC OpenEphys stream carrying the LED sync pulses.
    raw_xml_path, processed_xml_path, nrs_path, lfp_file_path : Path
        Neuroscope/OpenEphys metadata and LFP file paths.
    meta_path : Path
        Path to the metadata xlsx file.
    metadata_file_path : Path
        Path to the cohort metadata YAML file.
    mat_path : Path
        Path to the ``Processed/Analysis`` directory (tracking, spikes).
    sleep_path : Path
        Path to the ``Processed/Sleep`` directory.
    cue_epochs_path : Path
        Path to ``CueEpochs.mat``.
    ttl_folder_path : Path
        Path to the epoch-2 OpenEphys TTL events folder (containing ``states.npy``/``timestamps.npy``).
    video_file_paths, timestamps_file_paths : list[Path]
        Per-experiment AVI video and Bonsai tracking CSV files.
    save_path : Path
        Directory to write the NWB file (and transcoded videos).
    stub_test : bool, optional
        If True, stub the raw ephys/LFP to 100 samples for a fast smoke test.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # LOAD METADATA
    raw_xml_data = nwb.io.read_xml(raw_xml_path)
    processed_xml_data = nwb.io.read_xml(processed_xml_path)
    lfp_sampling_rate = float(raw_xml_data["eeg_sampling_rate"])
    nrs_data = nwb.io.read_nrs(nrs_path)
    metadata = nwb.io.read_metadata(meta_path, metadata_lookup_name, print_output=True)
    # The xlsx key uses an underscore (H6813_240605) but the NWB identifier needs the dash form
    # (H6813-240605) so create_nwb_file's split('-') recovers subject_id and session_id.
    metadata["file"]["name"] = folder_name
    metadata_from_yaml = load_dict_from_file(metadata_file_path)
    metadata = dict_deep_update(metadata, metadata_from_yaml)

    probe_info = get_probe_info_duszkiewicz(metadata)

    # CROSS-EXPERIMENT TIMEBASE: recover absolute Software Times → per-experiment sync offsets.
    software_start_ms = [
        nwb.io.read_openephys_software_start_ms(record_node_path / experiment_name / "recording1" / "sync_messages.txt")
        for experiment_name in experiment_names
    ]
    sync_offsets = [(start_ms - software_start_ms[0]) / 1000.0 for start_ms in software_start_ms]
    start_time = datetime.fromtimestamp(software_start_ms[0] / 1000.0, tz=pytz.timezone("Europe/London"))

    # LOAD PROCESSED DATA (no-gap MATLAB basis; aligned to NWB basis inside the adders)
    matlab_file_path = next(mat_path.glob("TrackingProcessed*.mat"))
    pos = nwb.io.get_matlab_position(matlab_file_path, vbl_name="pos")
    hd = nwb.io.get_matlab_hd(matlab_file_path, vbl_name="ang")
    spikes, waveforms, shank_id = nwb.io.get_matlab_spikes(mat_path)
    cue_epochs, blink_times = nwb.io.get_cue_epochs(cue_epochs_path)

    # VIDEO ALIGNMENT (LED TTL on ADC channel, per experiment, then shifted by sync offset)
    adc_channel = f"ADC{int(metadata['channelTTL']['0base']) - 128 + 1}"
    all_aligned_video_timestamps = get_aligned_video_timestamps_duszkiewicz(
        timestamp_file_paths=timestamps_file_paths,
        record_node_path=record_node_path,
        ttl_stream_name=ttl_stream_name,
        experiment_names=experiment_names,
        sync_offsets=sync_offsets,
        ttl_channel_id=adc_channel,
    )

    # CONSTRUCT NWB FILE
    nwbfile = nwb.convert.create_nwb_file(metadata, start_time)
    nwbfile, raw_xml_data = nwb.convert.add_probes(nwbfile, metadata, raw_xml_data, nrs_data, probe_info)
    nwbfile = nwb.convert.add_raw_ephys_multi_experiment(
        nwbfile=nwbfile,
        record_node_path=record_node_path,
        stream_name=stream_name,
        experiment_names=experiment_names,
        sync_offsets=sync_offsets,
        xml_data=raw_xml_data,
        stub_test=stub_test,
    )
    nwbfile = nwb.convert.add_lfp(nwbfile=nwbfile, lfp_path=lfp_file_path, xml_data=raw_xml_data, raw_eseries=nwbfile.acquisition["e-series"], stub_test=stub_test)
    lfp_eseries = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
    nwbfile = nwb.convert.add_tracking(nwbfile, pos, lfp_eseries, lfp_sampling_rate, ang=hd)
    nwbfile, camera_device = nwb.convert.add_camera_device(nwbfile=nwbfile, metadata=metadata)
    converted_video_file_paths = nwb.video_codec.convert_avi_to_mp4_h264(video_file_paths=video_file_paths, output_directory=save_path)
    # Store the video external_file paths relative to the NWB file (they are written alongside it),
    # as required by NWB best practices / DANDI.
    relative_video_file_paths = [Path(path.name) for path in converted_video_file_paths]
    nwbfile = nwb.convert.add_video(nwbfile=nwbfile, video_file_paths=relative_video_file_paths, all_aligned_video_timestamps=all_aligned_video_timestamps, metadata=metadata, camera_device=camera_device)
    nwbfile = nwb.convert.add_raw_tracking(nwbfile=nwbfile, file_paths=timestamps_file_paths, all_aligned_timestamps=all_aligned_video_timestamps, metadata=metadata, is_adult=True)
    nwbfile = nwb.convert.add_sleep(nwbfile, sleep_path, folder_name, lfp_eseries, lfp_sampling_rate)
    epochs = nwb.convert.get_epochs_from_eseries(eseries=nwbfile.acquisition["e-series"])
    nwbfile = nwb.convert.add_epochs(nwbfile, epochs, metadata)
    nwbfile = nwb.convert.add_units(nwbfile, raw_xml_data, processed_xml_data, spikes, waveforms, shank_id, lfp_eseries, lfp_sampling_rate)

    # Duszkiewicz-specific: processed cue events, blink timestamps, and raw DIO TTL events.
    nwbfile = nwb.convert.add_cue_events(nwbfile, cue_epochs, lfp_eseries, lfp_sampling_rate)
    nwbfile = nwb.convert.add_blink_events(nwbfile, blink_times, lfp_eseries, lfp_sampling_rate)
    ttl_experiment_name = next(name for name in experiment_names if name in ttl_folder_path.parts)
    dio_sync_offset = sync_offsets[experiment_names.index(ttl_experiment_name)]
    nwbfile = nwb.convert.add_dio_ttl_events(nwbfile, ttl_folder_path, dio_sync_offset)

    # save NWB file
    nwb.convert.save_nwb_file(nwbfile, save_path, folder_name)
