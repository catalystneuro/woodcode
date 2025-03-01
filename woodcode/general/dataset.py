import fmatch
from pathlib import Path
from typing import List, Dict, Union, Any
import warnings
from pynapple.io.interface_nwb import NWBFile
import h5py
import numpy as np
import pandas as pd
import os
import pynapple as nap

# Filter the specific HDMF warning about cached namespace
warnings.filterwarnings('ignore', message='Ignoring cached namespace.*', category=UserWarning)


def get_cell_metadata(nwb_files, metadata_fields=None):
    """
    Extract custom metadata from multiple NWB files with one row per cell.
    Column names will use only the last part of the field path.

    Parameters
    ----------
    nwb_files : list
        List of loaded NWB files
    metadata_fields : list, optional
        List of metadata fields to extract, specified as strings with dot notation
        e.g. ['lab', 'subject.genotype', 'subject.age']
    """
    if isinstance(nwb_files, NWBFile):
        nwb_files = [nwb_files]  # Store in a list if just a single nwb file

    recording_ids = [nwb.name for nwb in nwb_files]

    if metadata_fields is None:
        raise ValueError("Please specify a list of nwb fields to extract (e.g. 'protocol' or 'subject.genotype')")


    all_metadata = []  # List to store DataFrames from each recording

    for file, rec_id in zip(nwb_files, recording_ids):
        nwb = file.nwb
        n_cells = len(nwb.units)

        # Basic metadata for each cell
        metadata = {
            'nwb_file': [rec_id] * n_cells,
        }

        # Extract custom metadata fields
        for field in metadata_fields:
            try:
                # Split field path and get the last part for column name
                parts = field.split('.')
                column_name = parts[-1]  # Only keep last part

                # Navigate through the object hierarchy
                value = nwb
                for part in parts:
                    value = getattr(value, part)

                # Add to metadata dictionary
                metadata[column_name] = [value] * n_cells

            except AttributeError:
                print(f"Warning: Field '{field}' not found in recording {rec_id}")
                column_name = parts[-1]
                metadata[column_name] = [None] * n_cells

        # Create DataFrame for this recording
        df = pd.DataFrame(metadata)
        all_metadata.append(df)  # Append the DataFrame to our list

    if all_metadata:
        return pd.concat(all_metadata, ignore_index=True)
    else:
        raise ValueError("No metadata could be extracted from the files")


def create_nwb_file_list(data_dir: str, output_file: str, recursive: bool = True) -> int:
    """
    Scan a directory for NWB files and create a .list file containing their full paths.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing NWB files
    output_file : str
        Path where the .list file should be saved
    recursive : bool, optional
        If True, search subdirectories for NWB files (default: False)

    Returns
    -------
    int
        Number of NWB files found and written to the list
    """
    data_dir = Path(data_dir).resolve()
    output_file = Path(output_file)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if recursive:
        nwb_files = list(data_dir.rglob("*.nwb"))
    else:
        nwb_files = list(data_dir.glob("*.nwb"))

    if not nwb_files:
        raise ValueError(f"No NWB files found in {data_dir}")

    absolute_paths = [str(f.resolve()) for f in nwb_files]
    absolute_paths.sort()

    with open(output_file, 'w') as f:
        f.write("# NWB files list - one file path per line\n")
        f.write("# Add # at the start of a line to exclude files from upload\n\n")
        for file_path in absolute_paths:
            f.write(f"{file_path}\n")

    return len(absolute_paths)


def load_nwb_files(file_list_path: str) -> List[NWBFile]:
    """
    Load multiple NWB files using Pynapple from a list file containing full paths.
    Each NWB file is stored as a dictionary and includes its folder path.

    Parameters
    ----------
    file_list_path : str
        Path to the text file containing full paths to NWB files (one per line)

    Returns
    -------
    List[dict]
        List of loaded Pynapple NWB file dictionaries with an added 'path' key

    Raises
    ------
    FileNotFoundError
        If file_list_path or any NWB file doesn't exist
    ValueError
        If file_list_path contains no valid file paths
    """
    file_list_path = Path(file_list_path)

    if not file_list_path.exists():
        raise FileNotFoundError(f"File list not found: {file_list_path}")

    # Read file paths from the list, skipping comments and empty lines
    with open(file_list_path, 'r') as f:
        file_paths = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]

    if not file_paths:
        raise ValueError(f"No valid file paths found in {file_list_path}")

    nwb_files = []
    skipped_files = []

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            skipped_files.append((file_path, "File not found"))
            continue

        if path.suffix.lower() != '.nwb':
            skipped_files.append((file_path, "Not an NWB file"))
            continue

        try:
            nwb_file = nap.load_file(str(path))
            nwb_file["path"] = str(path.parent)  # Add folder path
            nwb_files.append(nwb_file)
        except Exception as e:
            skipped_files.append((file_path, str(e)))

    # Report any skipped files
    if skipped_files:
        print("\nWarning: Some files were skipped:")
        for file_path, reason in skipped_files:
            print(f"- {file_path}: {reason}")

    if not nwb_files:
        raise ValueError("No NWB files were successfully loaded")

    print(f"{len(nwb_files)} NWB files loaded")
    return nwb_files


from typing import List, Union
from pynwb import NWBFile

import fnmatch
from typing import List, Union, Any
from pynapple import NWBFile  # Assuming NWBFile comes from Pynapple

def filter_nwb_files(nwb_files: List[NWBFile], key: str, values: Union[Any, List[Any]]) -> List[NWBFile]:
    """
    Filter NWB files based on a specified metadata key-value pair or multiple possible values,
    allowing wildcard matching.

    Parameters
    ----------
    nwb_files : List[NWBFile]
        List of NWB files loaded using Pynapple.
    key : str
        The metadata key to check (e.g., "subject.genotype").
    values : Any or List[Any]
        The value(s) that the key must match for the NWB file to be included.
        Supports wildcards (e.g., 'spin*' matches 'spin1', 'spin2').

    Returns
    -------
    List[NWBFile]
        A list of NWB files that meet the filtering criterion.
    """

    if isinstance(nwb_files, NWBFile):
        nwb_files = [nwb_files]  # Store in a list if just a single NWB file

    if not isinstance(values, (list, tuple)):
        values = [values]  # Convert a single value to a list for easier comparison

    filtered_files = []

    for nwb_file in nwb_files:
        # Navigate nested metadata keys (e.g., "subject.genotype")
        keys = key.split(".")
        metadata = nwb_file.nwb  # Access the actual NWB structure from Pynapple

        try:
            for k in keys:
                metadata = getattr(metadata, k)  # Use attribute access

            # Check for exact match OR wildcard match
            if any(fnmatch.fnmatch(str(metadata), str(val)) for val in values):
                filtered_files.append(nwb_file)

        except AttributeError:
            # Skip if key is missing or not structured as expected
            continue

    print(f"{len(filtered_files)} files match the criterion: {key} in {values}")
    return filtered_files


import os
import h5py
import numpy as np
import pandas as pd
import pynapple as nap


def _save_to_hdf(parent_group, key, value):
    """
    Recursively save `value` into the HDF5 group `parent_group` under name `key`.
    Handles dictionary, numpy, pandas, and pynapple objects.
    """

    # If key already exists in parent_group, delete it to avoid conflicts
    if key in parent_group:
        del parent_group[key]

    # 1) Dictionary
    if isinstance(value, dict):
        # Create a new subgroup for this dict
        subgroup = parent_group.create_group(key)
        subgroup.attrs["type"] = "dict"
        # Recursively save each key-value pair
        for subkey, subval in value.items():
            _save_to_hdf(subgroup, subkey, subval)

    # 2) Numpy array
    elif isinstance(value, np.ndarray):
        ds = parent_group.create_dataset(key, data=value)
        ds.attrs["type"] = "numpy"

    # 3) Pandas DataFrame
    elif isinstance(value, pd.DataFrame):
        group = parent_group.create_group(key)
        group.attrs["type"] = "pandas_dataframe"
        group.create_dataset("values", data=value.to_numpy())
        group.attrs["columns"] = list(value.columns)
        group.attrs["index"] = list(value.index)

    # 4) Pandas Series
    elif isinstance(value, pd.Series):
        group = parent_group.create_group(key)
        group.attrs["type"] = "pandas_series"
        group.create_dataset("values", data=value.to_numpy())
        group.attrs["index"] = list(value.index)
        group.attrs["dtype"] = str(value.dtype)

    # 5) Pynapple Ts
    elif isinstance(value, nap.Ts):
        group = parent_group.create_group(key)
        group.attrs["type"] = "pynapple_ts"
        group.create_dataset("timestamps", data=value.as_units("s").values)

    # 6) Pynapple Tsd
    elif isinstance(value, nap.Tsd):
        group = parent_group.create_group(key)
        group.attrs["type"] = "pynapple_tsd"
        group.create_dataset("timestamps", data=value.index.as_units("s").values)
        group.create_dataset("values", data=value.values)

    # 7) Pynapple TsdFrame
    elif isinstance(value, nap.TsdFrame):
        group = parent_group.create_group(key)
        group.attrs["type"] = "pynapple_tsdframe"
        group.create_dataset("timestamps", data=value.index.as_units("s").values)
        group.create_dataset("values", data=value.to_numpy())
        group.attrs["columns"] = list(value.columns)

    # 8) Pynapple IntervalSet
    elif isinstance(value, nap.IntervalSet):
        group = parent_group.create_group(key)
        group.attrs["type"] = "pynapple_intervalset"
        group.create_dataset("start_times", data=value.start.as_units("s").values)
        group.create_dataset("end_times", data=value.end.as_units("s").values)

    else:
        raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")


def save_analysis(file_path, file_name, mode="w", skip_keys=None, **kwargs):
    """
    Save NumPy arrays, pandas DataFrames, pandas Series, Pynapple objects,
    and now dictionaries (possibly nested) into an HDF5 file.

    Parameters:
    - file_path (str): Directory where the HDF5 file should be saved.
    - file_name (str): Name of the HDF5 file (without extension).
    - mode (str): "w" to create/overwrite a file, "a" to append to an existing file
                  (overwrites keys by default).
    - skip_keys (list, optional): List of keys to skip overwriting in append mode.
    - kwargs: Named datasets (e.g., my_array=data, my_df=df, my_series=series, my_dict=some_dict).
    """
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Construct the full path for the HDF5 file
    file_name = file_name + '.h5'
    full_path = os.path.join(file_path, file_name)

    with h5py.File(full_path, mode) as f:
        if mode == "a":
            if skip_keys is None:
                skip_keys = []  # Default: No skipping, overwrite everything

        for key, value in kwargs.items():
            if mode == "a" and key in f and key in skip_keys:
                print(f"Skipping existing key '{key}' (as requested).")
                continue
            elif mode == "a" and key in f:
                # Remove the existing key to overwrite it
                del f[key]

            _save_to_hdf(f, key, value)

    print(f"Saved data to {full_path} (mode={mode})")


def _load_from_hdf(hdf_obj):
    """
    Recursively load a Python object from an HDF5 object (which can be either
    a Group or a Dataset). Returns the corresponding Python object.
    """
    # If hdf_obj is a dataset, we can read directly
    if isinstance(hdf_obj, h5py.Dataset):
        data_type = hdf_obj.attrs.get("type", None)

        if data_type == "numpy":
            return hdf_obj[()]  # returns a NumPy array
        else:
            raise ValueError(f"Unrecognized dataset type: {data_type}")

    else:
        # hdf_obj is a group
        group_type = hdf_obj.attrs.get("type", None)

        # 1) Dictionary
        if group_type == "dict":
            loaded_dict = {}
            for key in hdf_obj.keys():
                loaded_dict[key] = _load_from_hdf(hdf_obj[key])
            return loaded_dict

        # 2) Pandas DataFrame
        elif group_type == "pandas_dataframe":
            values = hdf_obj["values"][()]
            columns = hdf_obj.attrs["columns"]
            index = hdf_obj.attrs["index"]
            # Construct DataFrame
            df = pd.DataFrame(values, index=index, columns=columns)
            return df

        # 3) Pandas Series
        elif group_type == "pandas_series":
            values = hdf_obj["values"][()]
            index = hdf_obj.attrs["index"]
            dtype = hdf_obj.attrs["dtype"]
            # Construct Series
            s = pd.Series(values, index=index, dtype=dtype)
            return s

        # 4) Pynapple Ts
        elif group_type == "pynapple_ts":
            timestamps = hdf_obj["timestamps"][()]
            return nap.Ts(t=timestamps, time_units="s")

        # 5) Pynapple Tsd
        elif group_type == "pynapple_tsd":
            timestamps = hdf_obj["timestamps"][()]
            values = hdf_obj["values"][()]
            return nap.Tsd(t=timestamps, d=values, time_units="s")

        # 6) Pynapple TsdFrame
        elif group_type == "pynapple_tsdframe":
            timestamps = hdf_obj["timestamps"][()]
            values = hdf_obj["values"][()]
            columns = hdf_obj.attrs["columns"]
            return nap.TsdFrame(t=timestamps, d=values, columns=columns, time_units="s")

        # 7) Pynapple IntervalSet
        elif group_type == "pynapple_intervalset":
            start_times = hdf_obj["start_times"][()]
            end_times = hdf_obj["end_times"][()]
            return nap.IntervalSet(start=start_times, end=end_times, time_units="s")

        else:
            # If there's no recognized type, it might be a group with sub-groups/datasets
            # but missing the "type" attribute. You can handle that if needed.
            raise ValueError(f"Unsupported or missing 'type' attribute in group: {hdf_obj.name}")


def load_analysis(file_path, file_name):
    """
    Load objects (NumPy, pandas, pynapple, or dictionaries thereof) from an HDF5 file.

    Returns a dictionary where keys are the top-level groups/datasets in the HDF5 file
    and values are the corresponding loaded Python objects.
    """
    file_name = file_name + '.h5'
    full_path = os.path.join(file_path, file_name)

    loaded_data = {}

    with h5py.File(full_path, "r") as f:
        for key in f.keys():
            loaded_data[key] = _load_from_hdf(f[key])

    print(f"Loaded data from {full_path}")
    return loaded_data
