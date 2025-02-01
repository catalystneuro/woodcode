from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytz
from scipy import io as spio


def get_events(datapath, foldername, time_offset=0):

    print('Importing events npy files...')
    # importing events
    states = np.load(datapath / foldername / 'Analysis' / 'states.npy')
    timestamps = np.load(datapath / foldername / 'Analysis' / 'timestamps.npy')
    events = pd.Series(states, index=timestamps+time_offset)
    events = events[events > 0] # we don't care about the end of TTL pulses

    return events


def get_metadata(datapath, metaname, foldername):
    """
    Reads metadata from an Excel file and filters it based on `foldername`.

    Args:
        datapath (Path or str): Path to the directory containing the metadata file.
        metaname (str): Name of the Excel metadata file.
        foldername (str): The folder name to filter metadata.

    Returns:
        dict: Metadata as a dictionary.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the metadata file is empty or no matching recordings are found.
    """

    print(f'Importing metadata from: {datapath / metaname}')

    # Ensure the file exists
    metadata_path = datapath / metaname
    if not metadata_path.exists():
        raise FileNotFoundError(f"Error: Metadata file '{metadata_path}' not found.")

    # Load the Excel file
    try:
        metadata_full = pd.read_excel(metadata_path)
    except Exception as e:
        raise ValueError(f"Error loading Excel file '{metadata_path}': {e}")

    # Check if the DataFrame is empty
    if metadata_full.empty:
        raise ValueError(f"Error: Metadata file '{metadata_path}' is empty.")

    # Ensure 'recording' column exists
    if "recording" not in metadata_full.columns:
        raise KeyError(
            f"Error: Column 'recording' not found in metadata file '{metadata_path}'. Available columns: {list(metadata_full.columns)}")

    # Filter based on foldername
    metadata_filtered = metadata_full[metadata_full["recording"].astype(str).str.contains(foldername, na=False)]

    # Check if filtering resulted in any data
    if metadata_filtered.empty:
        raise ValueError(
            f"Error: No metadata found for foldername '{foldername}' in '{metadata_path}'. Available recordings: {metadata_full['recording'].dropna().unique()}")

    metadata_dict = metadata_filtered.iloc[0].to_dict()
    print("Metadata successfully imported.")

    return metadata_dict


def get_start_time(datapath, foldername):
    print('Importing start time from txt file...')
    # Read and extract the timestamp
    file_path = datapath / foldername / 'Analysis' / "Metadata.txt"
    with open(file_path, "r") as file:
        for line in file:
            if "Software Time" in line:
                # Extract the numeric part of the timestamp using split and indexing
                timestamp_str = line.split(":")[1].split()[0]  # Get the first part after the colon
                timestamp = int(timestamp_str) / 1000.0  # Milliseconds since epoch
                break
    # Convert to datetime and localize
    utc_datetime = datetime.utcfromtimestamp(timestamp)
    start_time = pytz.timezone('Europe/London').localize(utc_datetime)
    print(f"Recording start: {start_time}")

    return start_time


def get_data_matlab(datapath, foldername):

    print('Importing data from mat files...')

    path = datapath / foldername / 'Analysis'
    # file names for spikes, angle, and epoch files
    spike_file = path / 'SpikeData.mat'
    tracking_file = path / 'TrackingProcessed.mat'  # has HD angle
    epoch_file = path / 'Epoch_TS.csv'
    waveform_file = path / 'Waveforms.mat'
    wfeatures_file = path / 'WaveformFeatures.mat'

    # load epochs from mat file
    epochs = pd.read_csv(epoch_file, header=None, names=['Start','End'])


    """
    Loads position and angle data from a MATLAB .mat file,
    ensuring correct format and dimensions.
    """
    try:
        # Try loading as an HDF5 (v7.3) file
        with h5py.File(tracking_file, 'r') as tracking_data:
            print("Detected HDF5 MAT file (v7.3)")

            # Extract and process angle data
            ang = tracking_data['ang']['data'][()].T.squeeze()
            ang = np.atleast_1d(ang % (2 * np.pi))  # Ensure 1D

            ang_index = tracking_data['ang']['t'][()].T.squeeze()
            ang_index = np.atleast_1d(ang_index).ravel()  # Ensure index is 1D

            # Convert to Pandas Series
            ang = pd.Series(ang, index=ang_index)

            # Extract and process position data
            pos = tracking_data['pos']['data'][()].T
            pos = pd.DataFrame(pos, columns=['X', 'Y'])

            pos_index = tracking_data['pos']['t'][()].T.squeeze()
            pos_index = np.atleast_1d(pos_index).ravel()  # Ensure index is 1D
            pos.index = pos_index


    except OSError:
        # If not an HDF5 file, load using scipy.io (v7 or earlier)
        print("Detected older MAT file (v7 or earlier), using scipy.io.loadmat()")
        tracking_data = spio.loadmat(tracking_file, simplify_cells=True)

        # Extract and process angle data
        ang = np.atleast_1d(tracking_data['ang']['data'].squeeze() % (2 * np.pi))
        ang_index = np.atleast_1d(tracking_data['ang']['t'].squeeze()).ravel()  # Ensure index is 1D
        ang = pd.Series(ang, index=ang_index)

        pos = np.array(tracking_data['pos']['data'])  # Extract inner array
        pos = pos.squeeze()  # Remove unnecessary dimensions
        pos = pd.DataFrame(pos, columns=['X', 'Y'])
        pos_index = np.atleast_1d(tracking_data['pos']['t'].squeeze()).ravel()
        pos.index = pos_index

    # Next lines load the spike data from the .mat file
    spikedata = spio.loadmat(spike_file, simplify_cells=True)
    total_cells = np.arange(0, len(spikedata['S']['C']))  # To find the total number of cells in the recording
    spikes = dict()  # For pynapple, this will be turned into a TsGroup of all cells' timestamps
    cell_df = pd.DataFrame(columns=['timestamps'])  # Dataframe for cells and their spike timestamps
    # Loop to assign cell timestamps into the dataframe and the cell_ts dictionary
    for cell in total_cells:  # give spikes
        timestamps = spikedata['S']['C'][cell]['tsd']['t']
        cell_df.loc[cell, 'timestamps'] = timestamps
        temp = {cell: timestamps}
        spikes.update(temp)

   # get shank and channel ID for cells
    shank_id = spikedata['shank']-1

    # get waveforms and waveform features
    waveforms = spio.loadmat(waveform_file, simplify_cells=True)
    waveforms = waveforms['meanWaveforms']
    wfeatures = spio.loadmat(wfeatures_file, simplify_cells=True)
    maxIx = wfeatures['maxIx']
    tr2pk = wfeatures['tr2pk']

    return pos, ang, epochs, spikes, shank_id, waveforms, maxIx, tr2pk
