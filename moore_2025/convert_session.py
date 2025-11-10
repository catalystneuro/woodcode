"""Primary script to convert each example session to NWB."""
from pathlib import Path
import pandas as pd
import shutil
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update
import woodcode.nwb as nwb

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
    # lfp_file_path: Path,
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
    stub_test : bool, optional
        Whether to stub data for testing, by default False
    is_adult : bool, optional
        Whether the subject is an adult or a juvenile, by default True
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    # load all metadata
    xml_data = nwb.io.read_xml(xml_path)  # load all ephys info from the xml file
    nrs_data = nwb.io.read_nrs(nrs_path)  # load faulty channel info from the nrs file (i.e. channels not shown in Neuroscope)
    metadata = nwb.io.read_metadata(meta_path, folder_name, print_output=True)  # Load all metadata from the xlsx file
    start_time = nwb.io.get_start_time(folder_name, Path(""))  # load start time from Metadata.txt file # TODO: Fix this fn to generate a full start datetime

    # Load tracking, epochs and spikes from Matlab files (mostly loaded as pynapple objects)
    pos = nwb.io.get_matlab_position(mat_path / 'TrackingProcessed_Final.mat', vbl_name='pos')
    hd = nwb.io.get_matlab_hd(mat_path / 'TrackingProcessed_Final.mat', vbl_name='ang')
    epochs = pd.read_csv(mat_path / 'Epoch_TS.csv', header=None, names=['Start', 'End'])
    spikes, waveforms, shank_id = nwb.io.get_matlab_spikes(mat_path)

    # Update metadata with info from metadata.yaml
    metadata_from_yaml = load_dict_from_file(metadata_file_path)
    metadata = dict_deep_update(metadata, metadata_from_yaml)

    # CONSTRUCT NWB FILE
    nwbfile = nwb.convert.create_nwb_file(metadata, start_time)
    if is_adult:
        metadata["probe"][0]["coordinates"] = [-7.64, 3.3, -1.7]
    else:
        metadata["probe"][0]["coordinates"] = [-6.3, 3.1, -1.6]
    nwbfile = nwb.convert.add_probes(nwbfile, metadata, xml_data, nrs_data)
    nwbfile = nwb.convert.add_tracking(nwbfile, pos, hd)
    nwbfile = nwb.convert.add_units(nwbfile, xml_data, spikes, waveforms, shank_id)  # get shank names from NWB file
    # nwbfile = nwb.convert.add_events(nwbfile, events)
    nwbfile = nwb.convert.add_epochs(nwbfile, epochs, metadata)
    nwbfile = nwb.convert.add_sleep(nwbfile, sleep_path, folder_name)
    nwbfile = nwb.convert.add_video(nwbfile=nwbfile, video_file_paths=video_file_paths, timestamp_file_paths=timestamps_file_paths, metadata=metadata)
    # nwbfile = nwb.convert.add_lfp(nwbfile=nwbfile, lfp_path=lfp_file_path, xml_data=xml_data, stub_test=stub_test) # TODO: add LFP back in once it has been shared
    nwbfile = nwb.convert.add_raw_ephys(nwbfile=nwbfile, folder_path=raw_ephys_folder_path, epochs=epochs, xml_data=xml_data, stub_test=stub_test)

    # TODO: figure out what these events are
    # events = nwb.io.get_openephys_events(mat_path / 'states.npy', mat_path / 'timestamps.npy', time_offset=epochs.at[len(epochs)-1, 'Start'], skip_first=16)  # load LED events
    # nwbfile = nwb.convert.add_events(nwbfile, events)

    # save NWB file
    nwb.convert.save_nwb_file(nwbfile, save_path, folder_name)


def main():
    """Define paths and convert example sessions to NWB."""
    stub_test = True
    dataset_path = Path('/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset')
    output_folder_path = Path('/Volumes/T7/CatalystNeuro/Spyglass/raw')
    if output_folder_path.exists():
        shutil.rmtree(output_folder_path)
    metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/metadata.yaml")

    # Example Juvenile Sessions
    juvenile_folder_path = dataset_path / "H3000_Juveniles"

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
    # lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
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
        # lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=metadata_file_path,
        stub_test=stub_test,
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
    # lfp_file_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
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
        # lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=metadata_file_path,
        stub_test=stub_test,
    )


if __name__ == "__main__":
    main()
