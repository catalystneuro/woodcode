"""Primary script to convert each example session to NWB."""
from pathlib import Path
import pandas as pd
import shutil
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
    save_path: Path
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
    save_path : Path
        Path to save the NWB file
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    # load all metadata
    xml_data = nwb.io.read_xml(xml_path)  # load all ephys info from the xml file
    nrs_data = nwb.io.read_nrs(nrs_path)  # load faulty channel info from the nrs file (i.e. channels not shown in Neuroscope)
    metadata = nwb.io.read_metadata(meta_path, folder_name, print_output=True)  # Load all metadata from the xlsx file
    start_time = nwb.io.get_start_time(folder_name, dataset_path / folder_name / 'Analysis' / 'Metadata.txt')  # load start time from Metadata.txt file

    # Load tracking, epochs and spikes from Matlab files (mostly loaded as pynapple objects)
    pos = nwb.io.get_matlab_position(mat_path / 'TrackingProcessed.mat', vbl_name='pos')
    hd = nwb.io.get_matlab_hd(mat_path / 'TrackingProcessed.mat', vbl_name='ang')
    epochs = pd.read_csv(mat_path / 'Epoch_TS.csv', header=None, names=['Start', 'End'])
    spikes, waveforms, shank_id = nwb.io.get_matlab_spikes(mat_path)
    events = nwb.io.get_openephys_events(mat_path / 'states.npy', mat_path / 'timestamps.npy', time_offset=epochs.at[len(epochs)-1, 'Start'], skip_first=16)  # load LED events

    # CONSTRUCT NWB FILE
    nwbfile = nwb.convert.create_nwb_file(metadata, start_time)    
    # nwbfile = nwb.convert.add_probes(nwbfile, metadata, xml_data, nrs_data)
    # nwbfile = nwb.convert.add_tracking(nwbfile, pos, hd)
    # nwbfile = nwb.convert.add_units(nwbfile, xml_data, spikes, waveforms, shank_id)  # get shank names from NWB file
    # nwbfile = nwb.convert.add_events(nwbfile, events)
    # nwbfile = nwb.convert.add_epochs(nwbfile, epochs, metadata)
    # nwbfile = nwb.convert.add_sleep(nwbfile, sleep_path, folder_name)

    # save NWB file
    nwb.convert.save_nwb_file(nwbfile, save_path, folder_name)


def main():
    """Define paths and convert example sessions to NWB."""
    dataset_path = Path('/Volumes/T7/CatalystNeuro/Dudchenko')
    output_folder_path = Path('/Volumes/T7/CatalystNeuro/Spyglass/raw')
    if output_folder_path.exists():
        shutil.rmtree(output_folder_path)

    # Example session
    folder_name = 'H7115-250618'
    xml_path = dataset_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = dataset_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    meta_path = dataset_path / 'CatalystNeuro_metadata.xlsx'  # path to metadata file
    mat_path = dataset_path / folder_name / "Processed" / 'Analysis'
    sleep_path = dataset_path / folder_name / "Processed" / 'Sleep'
    save_path = output_folder_path

    session_to_nwb(
        dataset_path=dataset_path,
        folder_name=folder_name,
        xml_path=xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
