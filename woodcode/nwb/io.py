from datetime import datetime
import h5py
import pytz
from scipy import io as spio
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import json
import re

def get_events(datapath, foldername, time_offset=0):

    print('Importing events npy files...')
    # importing events
    states = np.load(datapath / foldername / 'Analysis' / 'states.npy')
    timestamps = np.load(datapath / foldername / 'Analysis' / 'timestamps.npy')
    events = pd.Series(states, index=timestamps+time_offset)
    events = events[events > 0]  # we don't care about the end of TTL pulses

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


def read_xml(file_path):
    """
    Parses an XML file and extracts relevant information into a dictionary.
    """
    tree = ET.parse(file_path)
    myroot = tree.getroot()

    data = {
        "nbits": int(myroot.find("acquisitionSystem").find("nBits").text),
        "dat_sampling_rate": None,
        "n_channels": None,
        "voltage_range": None,
        "amplification": None,
        "offset": None,
        "eeg_sampling_rate": None,
        "anatomical_groups": [],
        "skipped_channels": [],
        "discarded_channels": [],
        "spike_detection": [],
        "units": [],
        "spike_groups": []
    }

    for sf in myroot.findall("acquisitionSystem"):
        data["dat_sampling_rate"] = int(sf.find("samplingRate").text)
        data["n_channels"] = int(sf.find("nChannels").text)
        data["voltage_range"] = float(sf.find("voltageRange").text)
        data["amplification"] = float(sf.find("amplification").text)
        data["offset"] = float(sf.find("offset").text)

    for val in myroot.findall("fieldPotentials"):
        data["eeg_sampling_rate"] = int(val.find("lfpSamplingRate").text)

    anatomical_groups, skipped_channels = [], []
    for x in myroot.findall("anatomicalDescription"):
        for y in x.findall("channelGroups"):
            for z in y.findall("group"):
                chan_group = []
                for chan in z.findall("channel"):
                    if int(chan.attrib.get("skip", 0)) == 1:
                        skipped_channels.append(int(chan.text))
                    chan_group.append(int(chan.text))
                if chan_group:
                    anatomical_groups.append(np.array(chan_group))

    if data["n_channels"] is not None:
        data["discarded_channels"] = np.setdiff1d(
            np.arange(data["n_channels"]), np.concatenate(anatomical_groups) if anatomical_groups else []
        )

    data["anatomical_groups"] = np.array(anatomical_groups, dtype="object")
    data["skipped_channels"] = np.array(skipped_channels)

    # Parse spike detection groups
    spike_groups = []
    for x in myroot.findall("spikeDetection/channelGroups"):
        for y in x.findall("group"):
            chan_group = [int(chan.text) for chan in y.find("channels").findall("channel")]
            if chan_group:
                spike_groups.append(np.array(chan_group))

    data["spike_groups"] = np.array(spike_groups, dtype="object")

    # Parse unit clusters
    for unit in myroot.findall("units/unit"):
        data["units"].append({
            "group": int(unit.find("group").text),
            "cluster": int(unit.find("cluster").text),
            "structure": unit.find("structure").text or "",
            "type": unit.find("type").text or "",
            "isolationDistance": unit.find("isolationDistance").text or "",
            "quality": unit.find("quality").text or "",
            "notes": unit.find("notes").text or ""
        })

    return data


def read_metadata(file_path, file_name, print_output=False):
    """
    Reads an Excel metadata file and structures it into a dictionary,
    grouping metadata fields based on column name prefixes.
    Only reads the row where 'file_name' matches the given file_name.
    """
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Filter the row where 'file_name' matches the specified value
    df = df[df['file_name'] == file_name]

    if df.empty:
        raise ValueError(f"No entry found for file_name: {file_name}")

    # Reset index to access first row
    df = df.reset_index(drop=True)

    # Initialize metadata dictionary
    metadata = {}
    probe_data = {}

    probe_pattern = re.compile(r"probe_(\d+)_(\w+)")

    for col in df.columns:
        match = probe_pattern.match(col)
        if match:
            probe_num, probe_key = match.groups()
            probe_num = int(probe_num)  # Convert to integer
            if probe_num not in probe_data:
                probe_data[probe_num] = {"id": probe_num}

            value = df[col].iloc[0]
            if pd.isna(value):
                value = None
            probe_data[probe_num][probe_key] = value
        else:
            parts = col.split("_", 1)  # Split on the first underscore
            if len(parts) == 2:
                group, key = parts
            else:
                group, key = "misc", parts[0]  # If no underscore, classify under 'misc'

            # Get the first row value
            value = df[col].iloc[0]
            if pd.isna(value):
                value = None

            # Convert NumPy types to Python types for JSON compatibility
            if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype, pd.Series, pd.DataFrame)):
                value = value.item()

            if group not in metadata:
                metadata[group] = {}
            metadata[group][key] = value

    # Convert probe_data dictionary to a list of probe dictionaries
    metadata["probe"] = list(probe_data.values())

    # Print metadata if print_output is True
    if print_output:
        print(json.dumps(metadata, indent=4, default=str))

    return metadata