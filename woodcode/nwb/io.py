from datetime import datetime
import h5py
import pytz
from scipy import io as spio
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import json
import re
import pynapple as nap


def get_openephys_events(datapath, foldername, time_offset=0, skip_first=0):
# to do: account for instances when TTL is up at epoch edges

    print('Importing events npy files...')
    # importing events
    states = np.load(datapath / foldername / 'Analysis' / 'states.npy')
    timestamps = np.load(datapath / foldername / 'Analysis' / 'timestamps.npy')
    events = pd.Series(states, index=timestamps+time_offset)
    events = events.iloc[skip_first:]

    df = events.reset_index()
    df.columns = ['time', 'event']

    # Identify start and end events
    df['event_type'] = df['event'].abs()
    df['event_sign'] = df['event'].apply(lambda x: 'start' if x > 0 else 'end')

    # Separate dataframes for each event type
    event_dict = {}
    for event_type in df['event_type'].unique():
        event_subset = df[df['event_type'] == event_type].copy()

        starts = event_subset[event_subset['event_sign'] == 'start'].reset_index(drop=True)
        ends = event_subset[event_subset['event_sign'] == 'end'].reset_index(drop=True)
        event_int = nap.IntervalSet(start=starts['time'].values, end=ends['time'].values)

        event_dict[event_type] = event_int

    return event_dict


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
    # to do: if file not found,try using recording name and midnight
    # if that fails, use placeholder time

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


def get_matlab_position(pos_file, vbl_name='pos'):
    """
    Loads position from a MATLAB .mat file,
    ensuring correct format and dimensions.
    """
    print('Importing position from mat file...')

    try:
        # Try loading as an HDF5 (v7.3) file
        with h5py.File(pos_file, 'r') as pos_data:
            # Extract and process position data
            pos = pos_data[vbl_name]['data'][()].T
            pos_index = pos_data[vbl_name]['t'][()].T.squeeze()
            pos_index = np.atleast_1d(pos_index).ravel()  # Ensure index is 1D

    except OSError:
        # If not an HDF5 file, load using scipy.io (v7 or earlier)
        pos_data = spio.loadmat(pos_file, simplify_cells=True)
        pos = np.array(pos_data[vbl_name]['data']).squeeze()  # Extract inner array and remove unnecessary dimensions
        pos_index = np.atleast_1d(pos_data['pos']['t'].squeeze()).ravel()

    pos_tsdframe = nap.TsdFrame(t=pos_index, d=pos, columns=['x', 'y'])

    return pos_tsdframe

def get_matlab_hd(hd_file, vbl_name='ang'):
    """
    Loads head-direction data from a MATLAB .mat file,
    ensuring correct format and dimensions.
    """
    print('Importing position from mat file...')

    try:
        # Try loading as an HDF5 (v7.3) file
        with h5py.File(hd_file, 'r') as hd_data:
            # Extract and process angle data
            hd = hd_data[vbl_name]['data'][()].T.squeeze()
            hd = np.atleast_1d(hd % (2 * np.pi))  # Ensure 1D
            hd_index = hd_data['ang']['t'][()].T.squeeze()
            hd_index = np.atleast_1d(hd_index).ravel()  # Ensure index is 1D

    except OSError:
        # If not an HDF5 file, load using scipy.io (v7 or earlier)
        hd_data = spio.loadmat(hd_file, simplify_cells=True)
        hd = np.array(hd_data[vbl_name]['data']).squeeze()  # Extract inner array
        hd_index = np.atleast_1d(hd_data['pos']['t'].squeeze()).ravel()

    hd_tsd = nap.Tsd(t=hd_index, d=hd)

    return hd_tsd

def get_matlab_spikes(path):

    print('Importing data from mat files...')

    # file names for spikes, angle, and epoch files
    spike_file = path / 'SpikeData.mat'
    waveform_file = path / 'Waveforms.mat'
    wfeatures_file = path / 'WaveformFeatures.mat'

    # Next lines load the spike data from the .mat file
    spikedata = spio.loadmat(spike_file, simplify_cells=True)
    total_cells = np.arange(0, len(spikedata['S']['C']))
    spikes = {}
    for cell in total_cells:
        timestamps = np.array(spikedata['S']['C'][cell]['tsd']['t'])  # Convert to numpy array
        spikes[cell] = nap.Ts(t=timestamps)  # Create Ts object for each neuron

    # get spike metadata
    waveforms = spio.loadmat(waveform_file, simplify_cells=True)
    waveforms = waveforms['meanWaveforms']
    wfeatures = spio.loadmat(wfeatures_file, simplify_cells=True)
    shank_id = spikedata['shank']-1,
    shank_id = shank_id[0]
    maxIx = wfeatures['maxIx']

    spikes = nap.TsGroup(spikes)  # Convert dictionary to TsGroup

    return spikes, waveforms, shank_id


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