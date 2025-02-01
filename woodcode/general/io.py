import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import json
import re


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

# Example usage
# metadata_dict = read_metadata("metadata.xlsx", "H6813-240606", print_output=True)

