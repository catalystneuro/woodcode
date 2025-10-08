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
    video_file_paths: list[Path],
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

    metadata = {
        'Video': {
            'ImageSeries': {
                'name': 'Video',
                'description': 'Video capturing and marker tracking were performed using Bonsai software 51 at 40 Hz. For the juvenile recordings, images were acquired through a Logitech C930e camera. To synchronise the images with the electrophysiology data, an Arduino microcontroller was programmed to send a random sequence of pulses to both the OpenEphys system and a LED light within the frame of the camera. The times when the LED light shone were detected by Bonsai and synchronisation of the pulses in the light and the electrophysiology data was done offline. For the adult recordings, images were acquired through an acA1300-75gc Basler camera with a LMVZ4411 Kowa lens, positioned at the ceiling of the rig. This camera was connected to the OpenEphys board, sending a pulse for each frame taken, allowing the synchronisation of the two streams of information.',
            },
            'CameraDevice': {
                'name': 'camera_device 0', # This MUST be formatted exactly "camera_device {camera_id}" to be compatible with spyglass
                'meters_per_pixel': 0.0016, # TODO: update this value
                'manufacturer': 'Logitech',
                'model': 'C930e',
                'lens': 'built-in',
                'camera_name': 'Video Camera',
            },
        }
    }
    nwbfile = nwb.convert.add_video(nwbfile=nwbfile, video_file_paths=video_file_paths, metadata=metadata, rate=40.0)

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
    video_file_paths = [
        dataset_path / folder_name / "Raw" / "2025-06-18_15-23-44" / "Record Node 101" / "experiment1" / "recording1" / "BonsaiVideo2025-06-18T15_23_50.avi",
        dataset_path / folder_name / "Raw" / "2025-06-18_15-23-44" / "Record Node 101" / "experiment1" / "recording2" / "BonsaiVideo2025-06-18T15_36_52.avi",
        dataset_path / folder_name / "Raw" / "2025-06-18_15-23-44" / "Record Node 101" / "experiment1" / "recording3" / "BonsaiVideo2025-06-18T17_10_04.avi",
    ]
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
        save_path=save_path
    )


if __name__ == "__main__":
    main()
