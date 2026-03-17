"""Primary script to convert each example session to NWB."""
from pathlib import Path
import shutil
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update
import woodcode.nwb as nwb
import probeinterface as pbi
import numpy as np

from moore_2025.temporal_alignment import get_aligned_video_timestamps_juveniles, get_aligned_video_timestamps_adults, get_start_time, get_unaligned_video_timestamps_juveniles

def get_probe_info_juveniles() -> dict:
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


DAT_FILE_TEMPORAL_ALIGNMENT_WARNING = (
    "WARNING: Ephys data was loaded from a raw .dat file because the Open Ephys output folder "
    "is unavailable for this session. Two temporal alignment issues apply to all time series in "
    "this file: (1) The ephys time basis is NOT aligned to the behavioral time basis — ephys and "
    "behavioral timestamps cannot be used to cross-reference signals. (2) The recording consists "
    "of multiple segments that were separated by real-world gaps (on the order of 5-10 minutes); "
    "those gaps are NOT represented in the timestamps — the segments are simply concatenated, so "
    "timestamps are discontinuous at segment boundaries without any indication of where breaks occur."
)


def session_to_nwb(
    *,
    folder_name: str,
    raw_xml_path: Path,
    processed_xml_path: Path,
    nrs_path: Path,
    meta_path: Path,
    mat_path: Path,
    sleep_path: Path,
    timestamps_file_paths: list[Path],
    lfp_file_path: Path,
    save_path: Path,
    metadata_file_path: Path,
    histology_folder_path: Path,
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
    timestamps_file_paths : list[Path]
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
    if has_open_ephys_output:
        assert raw_ephys_dat_file_path is None, "Cannot provide both raw_ephys_folder_path and raw_ephys_dat_file_path"
    else:
        assert raw_ephys_dat_file_path is not None, "Must provide either raw_ephys_folder_path or raw_ephys_dat_file_path"
    if is_adult:
        probe_info = get_probe_info_adults()
        if has_video:
            all_aligned_video_timestamps = get_aligned_video_timestamps_adults(
                timestamp_file_paths=timestamps_file_paths,
                ephys_folder_path=raw_ephys_folder_path,
                ttl_stream_name=ttl_stream_name,
            )
    else: # juvenile
        probe_info = get_probe_info_juveniles()
        if has_video:
            if has_open_ephys_output:
                aligned_video_timestamps = get_aligned_video_timestamps_juveniles(
                    timestamp_file_path=timestamps_file_paths[0],
                    ephys_folder_path=raw_ephys_folder_path,
                    ttl_stream_name=ttl_stream_name,
                )
                all_aligned_video_timestamps = [aligned_video_timestamps]
            else:
                unaligned_video_timestamps = get_unaligned_video_timestamps_juveniles(timestamp_file_path=timestamps_file_paths[0])
                all_aligned_video_timestamps = [unaligned_video_timestamps]

    save_path.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    # load all metadata
    raw_xml_data = nwb.io.read_xml(raw_xml_path)  # load all ephys info from the xml file
    processed_xml_data = nwb.io.read_xml(processed_xml_path)  # load all ephys info from the processed xml file
    lfp_sampling_rate = float(raw_xml_data['eeg_sampling_rate'])
    nrs_data = nwb.io.read_nrs(nrs_path)  # load faulty channel info from the nrs file (i.e. channels not shown in Neuroscope)
    first_video_file_path = video_file_paths[0] if has_video else None
    metadata = nwb.io.read_metadata(meta_path, folder_name, print_output=True)  # Load all metadata from the xlsx file
    start_time = get_start_time(timestamps_file_paths[0], first_video_file_path)  # load start time from the first timestamps CSV file

    # Load tracking, epochs and spikes from Matlab files (mostly loaded as pynapple objects)
    pos = nwb.io.get_matlab_position(mat_path / 'TrackingProcessed_Final.mat', vbl_name='pos')
    hd = nwb.io.get_matlab_hd(mat_path / 'TrackingProcessed_Final.mat', vbl_name='ang')
    spikes, waveforms, shank_id = nwb.io.get_matlab_spikes(mat_path)

    # Update metadata with info from metadata.yaml
    metadata_from_yaml = load_dict_from_file(metadata_file_path)
    metadata = dict_deep_update(metadata, metadata_from_yaml)

    # CONSTRUCT NWB FILE
    dat_file_comments = DAT_FILE_TEMPORAL_ALIGNMENT_WARNING if not has_open_ephys_output else "no comments"
    nwbfile = nwb.convert.create_nwb_file(metadata, start_time)
    nwbfile, raw_xml_data = nwb.convert.add_probes(nwbfile, metadata, raw_xml_data, nrs_data, probe_info)
    if has_open_ephys_output:
        nwbfile = nwb.convert.add_raw_ephys(nwbfile=nwbfile, folder_path=raw_ephys_folder_path, xml_data=raw_xml_data, stream_name=stream_name, stub_test=stub_test)
    else:
        nwbfile = nwb.convert.add_raw_ephys_from_dat(nwbfile=nwbfile, dat_file_path=raw_ephys_dat_file_path, xml_data=raw_xml_data, stub_test=stub_test, comments=dat_file_comments)
    nwbfile = nwb.convert.add_lfp(nwbfile=nwbfile, lfp_path=lfp_file_path, xml_data=raw_xml_data, raw_eseries=nwbfile.acquisition['e-series'], stub_test=stub_test, comments=dat_file_comments)
    lfp_eseries = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
    nwbfile = nwb.convert.add_tracking(nwbfile, pos, lfp_eseries, lfp_sampling_rate, ang=hd, comments=dat_file_comments)
    if has_video:
        nwbfile = nwb.convert.add_video(nwbfile=nwbfile, video_file_paths=video_file_paths, all_aligned_video_timestamps=all_aligned_video_timestamps, metadata=metadata, comments=dat_file_comments)
        nwbfile = nwb.convert.add_raw_tracking(nwbfile=nwbfile, file_paths=timestamps_file_paths, all_aligned_timestamps=all_aligned_video_timestamps, metadata=metadata, is_adult=is_adult, comments=dat_file_comments)
    nwbfile = nwb.convert.add_sleep(nwbfile, sleep_path, folder_name, lfp_eseries, lfp_sampling_rate, comments=dat_file_comments)
    epochs = nwb.convert.get_epochs_from_eseries(eseries=nwbfile.acquisition['e-series'])
    nwbfile = nwb.convert.add_epochs(nwbfile, epochs, metadata)
    nwbfile = nwb.convert.add_units(nwbfile, raw_xml_data, processed_xml_data, spikes, waveforms, shank_id, lfp_eseries, lfp_sampling_rate)  # get shank names from NWB file
    nwbfile = nwb.convert.add_histology(nwbfile, histology_folder_path)

    # save NWB file
    nwb.convert.save_nwb_file(nwbfile, save_path, folder_name)


def main():
    """Define paths and convert example sessions to NWB."""
    juvenile_folder_path = Path("/Volumes/SamsungSSD/CatalystNeuro/Dudchenko/251104_MooreDataset/H3000_Juveniles")
    adult_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/H4800_Adults")
    meta_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/MooreDataset_Metadata.xlsx")
    histology_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/Histology")
    output_folder_path = Path('/Volumes/T7/CatalystNeuro/Spyglass/raw')
    juvenile_metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/juvenile_metadata.yaml")
    adult_metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/adult_metadata.yaml")

    stub_test = True
    if output_folder_path.exists():
        shutil.rmtree(output_folder_path)
    save_path = output_folder_path

    # Example Juvenile Sessions
    juvenile_histology_folder_path = histology_folder_path / "H3000"

    # Example Juvenile WT sessions
    jv_wt_folder_path = juvenile_folder_path / "WT"

    # Day 1
    folder_name = 'H3022-210805'
    raw_xml_path = jv_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
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

    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stream_name="Rhythm_FPGA-100.0",
        ttl_stream_name="Rhythm_FPGA-100.0_ADC",
        stub_test=stub_test,
        is_adult=False,
    )

    # Day 2
    folder_name = 'H3022-210806'
    raw_xml_path = jv_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-08-06T11_34_08.avi",
    ]
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "Bonsai testing2021-08-06T11_34_07.csv",
    ]
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_wt_folder_path / folder_name / "Raw"
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stream_name="Rhythm_FPGA-100.0",
        ttl_stream_name="Rhythm_FPGA-100.0_ADC",
        stub_test=stub_test,
        is_adult=False,
    )

    # Example Juvenile KO sessions
    jv_ko_folder_path = juvenile_folder_path / "KO"

    # Day 1
    folder_name = 'H3016-210422'
    raw_xml_path = jv_ko_folder_path / folder_name / "Raw" / "experiment3" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.xml')
    nrs_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_ko_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_ko_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-04-22T18_15_24.avi",
    ]
    timestamps_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "Bonsai testing2021-04-22T18_15_24.csv",
    ]
    lfp_file_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_ko_folder_path / folder_name / "Raw"
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stream_name="Rhythm_FPGA-100.0",
        ttl_stream_name="Rhythm_FPGA-100.0_ADC",
        stub_test=stub_test,
        is_adult=False,
    )

    # Day 2
    folder_name = 'H3016-210423'
    raw_xml_path = jv_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.xml')
    nrs_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
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
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stream_name="Rhythm_FPGA-100.0",
        ttl_stream_name="Rhythm_FPGA-100.0_ADC",
        stub_test=stub_test,
        is_adult=False,
    )

    # Example Adult Sessions
    adult_histology_folder_path = histology_folder_path / "H4800"

    # Example Adult WT session
    adult_wt_folder_path = adult_folder_path / "WT"
    folder_name = 'H4813-220728'
    # Note: using the XML from the Raw folder here since the one in Processed is missing one of the channels for shank 2
    processed_xml_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    raw_xml_path = adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-103.0" / "continuous.xml"
    nrs_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
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
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=adult_metadata_file_path,
        histology_folder_path=adult_histology_folder_path,
        stream_name="Rhythm_FPGA-103.0",
        ttl_stream_name="Rhythm_FPGA-103.0_ADC",
        stub_test=stub_test,
        is_adult=True,
    )

    # Example Adult KO session
    adult_ko_folder_path = adult_folder_path / "KO"
    folder_name = 'H4817-220828'
    # Raw XML for this session is missing one of the channels (channel 38 on shank 1), so using the Processed XML instead
    processed_xml_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    raw_xml_path = processed_xml_path
    nrs_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
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
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=adult_metadata_file_path,
        histology_folder_path=adult_histology_folder_path,
        stream_name="Rhythm_FPGA-103.0",
        ttl_stream_name="Rhythm_FPGA-103.0_ADC",
        stub_test=stub_test,
        is_adult=True,
    )

    # Edge Case Sessions
    # Example Session without videos
    jv_wt_folder_path = juvenile_folder_path / "WT"
    juvenile_histology_folder_path = histology_folder_path / "H3000"
    folder_name = 'H3001-200202'
    raw_xml_path = jv_wt_folder_path / folder_name / "Raw" / "H3001-200202" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "H3001-200202" / "experiment1" / "Bonsai testing2020-02-02T18_27_37.csv",
    ]
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_wt_folder_path / folder_name / "Raw" / "H3001-200202"
    save_path = output_folder_path
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stream_name="Rhythm_FPGA-100.0",
        ttl_stream_name="Rhythm_FPGA-100.0_ADC",
        stub_test=stub_test,
        is_adult=False,
    )

    # Example Session without raw data
    jv_wt_folder_path = juvenile_folder_path / "WT"
    juvenile_histology_folder_path = histology_folder_path / "H3000"
    folder_name = 'H3023-210812'
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    raw_xml_path = processed_xml_path  # Raw data for this session is missing, so using the Processed XML instead
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_wt_folder_path / folder_name / "Processed" / "BonsaiCaptureALL2021-08-12T19_44_12.avi",
    ]
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Processed" / "BonsaiTracking.csv",
    ]
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_dat_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.dat')  # Raw .dat file for this session since raw OpenEphys folder is missing
    save_path = output_folder_path
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_dat_file_path=raw_ephys_dat_file_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stub_test=stub_test,
        is_adult=False,
    )

    # # TODO: add this session once the .nrs file has been shared
    # # Example Juvenile Session with Adult temporal alignment and H5 Probe
    # jv_wt_folder_path = juvenile_folder_path / "WT"
    # juvenile_histology_folder_path = histology_folder_path / "H3000"
    # folder_name = 'H3029-230510'
    # processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    # raw_xml_path = jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "recording1" / "continuous" / "Acquisition_Board-100.Rhythm Data" / "continuous.xml"
    # nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    # mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    # sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    # video_file_paths = [
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T12_12_47.avi",
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T12_34_44.avi",
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T14_07_30.avi",
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T14_31_02.avi",
    # ]
    # timestamps_file_paths = [
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T12_12_45.csv",
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T12_34_42.csv",
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T14_07_28.csv",
    #     jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T14_31_01.csv",
    # ]
    # lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    # raw_ephys_folder_path = jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2"
    # save_path = output_folder_path
    # session_to_nwb(
    #     folder_name=folder_name,
    #     raw_xml_path=raw_xml_path,
    #     processed_xml_path=processed_xml_path,
    #     nrs_path=nrs_path,
    #     meta_path=meta_path,
    #     mat_path=mat_path,
    #     sleep_path=sleep_path,
    #     video_file_paths=video_file_paths,
    #     timestamps_file_paths=timestamps_file_paths,
    #     lfp_file_path=lfp_file_path,
    #     raw_ephys_folder_path=raw_ephys_folder_path,
    #     save_path=save_path,
    #     metadata_file_path=juvenile_metadata_file_path,
    #     histology_folder_path=juvenile_histology_folder_path,
    #     stream_name="Acquisition_Board-100.Rhythm Data",
    #     ttl_stream_name="Acquisition_Board-100.Rhythm Data_ADC",
    #     stub_test=stub_test,
    #     is_adult=True,
    # )

    # Example Adult Session with error epoch
    adult_wt_folder_path = adult_folder_path / "WT"
    adult_histology_folder_path = histology_folder_path / "H4800"
    folder_name = 'H4830-230406'
    raw_xml_path = adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "Record Node 103" / "experiment1" / "recording1" / "continuous" / "Acquisition_Board-100.Rhythm Data" / "continuous.xml"
    processed_xml_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.xml')  # path to xml file
    nrs_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.nrs')  # path to xml file
    mat_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / 'Analysis'
    sleep_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / 'Sleep'
    video_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T19_28_42.avi",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T21_15_19.avi",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T21_39_18.avi",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T21_49_57.avi",
    ]
    timestamps_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T19_05_55.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T19_28_41.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T21_15_17.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T21_39_17.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T21_49_56.csv",
    ]
    lfp_file_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.lfp')
    raw_ephys_folder_path = adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52"
    save_path = output_folder_path
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=adult_metadata_file_path,
        histology_folder_path=adult_histology_folder_path,
        stream_name="Record Node 103#Acquisition_Board-100.Rhythm Data",
        ttl_stream_name="Record Node 103#Acquisition_Board-100.Rhythm Data_ADC",
        stub_test=stub_test,
        is_adult=True,
    )

if __name__ == "__main__":
    main()
