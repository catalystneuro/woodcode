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
    # read metadata from xml file
    print('Importing metadata from xml file... ')
    metadata_full = pd.read_excel(datapath / metaname)
    metadata = metadata_full[metadata_full["recording"].str.contains(foldername)]
    metadata = metadata.iloc[0].to_dict()

    return metadata


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

    # Load angle from mat file
    tracking_data = h5py.File(tracking_file, 'r')
    ang = tracking_data['ang']['data'][()].T.squeeze()
    ang = ang % (2 * np.pi)  # mod 2 pi
    ang = pd.Series(ang)
    ang.index = tracking_data['ang']['t'][()].T.squeeze()

    # load position from mat file
    pos = tracking_data['pos']['data'][()].T
    pos = pd.DataFrame(pos,columns=['X', 'Y'])
    pos.index = tracking_data['pos']['t'][()].T.squeeze()

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
