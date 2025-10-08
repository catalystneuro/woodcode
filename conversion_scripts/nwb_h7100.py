from pathlib import Path
import pandas as pd
import woodcode.nwb as nwb

def main():
    dataset_path = Path('/Volumes/T7/CatalystNeuro/Dudchenko/NWB_Conversion')
    folder_name = 'H7115-250618'
    xml_path = dataset_path / folder_name / (folder_name + '.xml')  # path to xml file
    nrs_path = dataset_path / folder_name / (folder_name + '.nrs')  # path to xml file
    meta_path = dataset_path / 'CatalystNeuro_metadata.xlsx'  # path to metadata file
    mat_path = dataset_path / folder_name / 'Analysis'
    sleep_path = dataset_path / folder_name / 'Sleep'
    save_path = dataset_path / 'NWB' / folder_name

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
    events = nwb.io.get_openephys_events(dataset_path / folder_name / 'Analysis' / 'states.npy', dataset_path / folder_name / 'Analysis' / 'timestamps.npy', time_offset=epochs.at[len(epochs)-1, 'Start'], skip_first=16)  # load LED events

    # CONSTRUCT NWB FILE
    nwbfile = nwb.convert.create_nwb_file(metadata, start_time)    
    # add probes
    nwbfile = nwb.convert.add_probes(nwbfile, metadata, xml_data, nrs_data)
    # add tracking
    nwbfile = nwb.convert.add_tracking(nwbfile, pos, hd)
    # add spikes
    nwbfile = nwb.convert.add_units(nwbfile, xml_data, spikes, waveforms, shank_id)  # get shank names from NWB file
    # Add event times
    nwbfile = nwb.convert.add_events(nwbfile, events)
    # Add epochs
    nwbfile = nwb.convert.add_epochs(nwbfile, epochs, metadata)
    # Add sleep scoring and pseudo EMG
    nwbfile = nwb.convert.add_sleep(nwbfile, sleep_path, folder_name)

    # save NWB file
    nwb.convert.save_nwb_file(nwbfile, save_path, folder_name)
    # load NWB file (for testing)
   # data = nwb.convert.load_nwb_file(save_path, folder_name)


main()
