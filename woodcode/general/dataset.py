
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

def filter_nwb_files(nwb_files: List[NWBFile], key: str, values: Union[Any, List[Any]]) -> List[NWBFile]:
    """
    Filter NWB files based on a specified metadata key-value pair or multiple possible values.

    Parameters
    ----------
    nwb_files : List[NWBFile]
        List of NWB files loaded using Pynapple.
    key : str
        The metadata key to check (e.g., "subject.genotype").
    values : Any or List[Any]
        The value(s) that the key must match for the NWB file to be included.
        Can be a single value or a list of values.

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
            if metadata in values:
                filtered_files.append(nwb_file)
        except AttributeError:
            # Skip if key is missing or not structured as expected
            continue

    print(f"{len(filtered_files)} files match the criterion: {key} in {values}")
    return filtered_files



def save_analysis(file_path, file_name, mode="w", skip_keys=None, **kwargs):
    """
    Save NumPy arrays, pandas DataFrames, pandas Series, and Pynapple objects (Ts, Tsd, TsdFrame, IntervalSet) into an HDF5 file.

    Parameters:
    - file_path (str): Directory where the HDF5 file should be saved.
    - file_name (str): Name of the HDF5 file (without extension).
    - mode (str): "w" to create/overwrite a file, "a" to append to an existing file (overwrites keys by default).
    - skip_keys (list, optional): List of keys to skip overwriting in append mode.
    - kwargs: Named datasets (e.g., my_array=data, my_df=df, my_series=series).
    """
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Construct the full path for the HDF5 file
    file_name = file_name + '.h5'
    full_path = os.path.join(file_path, file_name)

    with h5py.File(full_path, mode) as f:
        if mode == "a":
            existing_keys = list(f.keys())
            if skip_keys is None:
                skip_keys = []  # Default: No skipping, overwrite everything

        for key, value in kwargs.items():
            if mode == "a" and key in skip_keys:
                print(f"Skipping existing key '{key}' (as requested).")
                continue
            elif key in f and mode == "a":
                del f[key]  # Remove the existing key to overwrite it

            # Store the dataset based on type
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
                f[key].attrs["type"] = "numpy"
            elif isinstance(value, pd.DataFrame):
                group = f.create_group(key)
                group.create_dataset("values", data=value.to_numpy())
                group.attrs["columns"] = list(value.columns)
                group.attrs["index"] = list(value.index)
                group.attrs["type"] = "pandas_dataframe"
            elif isinstance(value, pd.Series):
                group = f.create_group(key)
                group.create_dataset("values", data=value.to_numpy())
                group.attrs["index"] = list(value.index)
                group.attrs["dtype"] = str(value.dtype)
                group.attrs["type"] = "pandas_series"
            elif isinstance(value, nap.Ts):
                group = f.create_group(key)
                group.create_dataset("timestamps", data=value.as_units("s").values)
                group.attrs["type"] = "pynapple_ts"
            elif isinstance(value, nap.Tsd):
                group = f.create_group(key)
                group.create_dataset("timestamps", data=value.index.as_units("s").values)
                group.create_dataset("values", data=value.values)
                group.attrs["type"] = "pynapple_tsd"
            elif isinstance(value, nap.TsdFrame):
                group = f.create_group(key)
                group.create_dataset("timestamps", data=value.index.as_units("s").values)
                group.create_dataset("values", data=value.to_numpy())
                group.attrs["columns"] = list(value.columns)
                group.attrs["type"] = "pynapple_tsdframe"
            elif isinstance(value, nap.IntervalSet):
                group = f.create_group(key)
                group.create_dataset("start_times", data=value.start.as_units("s").values)
                group.create_dataset("end_times", data=value.end.as_units("s").values)
                group.attrs["type"] = "pynapple_intervalset"
            else:
                raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")

    print(f"Saved data to {full_path} (mode={mode})")


def load_analysis(file_path, file_name):
    """
    Load NumPy arrays, pandas DataFrames, pandas Series, and Pynapple objects (Ts, Tsd, TsdFrame, IntervalSet) from an HDF5 file.

    Parameters:
    - file_path (str): Directory where the HDF5 file is located.
    - file_name (str): Name of the HDF5 file (without extension).

    Returns:
    - dict: A dictionary containing loaded objects.
    """
    file_name = file_name + '.h5'
    full_path = os.path.join(file_path, file_name)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File '{full_path}' not found.")

    data = {}
    with h5py.File(full_path, "r") as f:
        for key in f.keys():
            obj_type = f[key].attrs.get("type", "unknown")

            if obj_type == "numpy":
                data[key] = f[key][:]
            elif obj_type == "pandas_dataframe":
                values = f[key]["values"][:]
                columns = list(f[key].attrs["columns"])
                index = list(f[key].attrs["index"])
                data[key] = pd.DataFrame(values, columns=columns, index=index)
            elif obj_type == "pandas_series":
                values = f[key]["values"][:]
                index = list(f[key].attrs["index"])
                dtype = f[key].attrs["dtype"]
                data[key] = pd.Series(values, index=index, dtype=dtype)
            elif obj_type == "pynapple_ts":
                timestamps = f[key]["timestamps"][:]
                data[key] = nap.Ts(t=timestamps, time_units="s")
            elif obj_type == "pynapple_tsd":
                timestamps = f[key]["timestamps"][:]
                values = f[key]["values"][:]
                data[key] = nap.Tsd(t=timestamps, d=values, time_units="s")
            elif obj_type == "pynapple_tsdframe":
                timestamps = f[key]["timestamps"][:]
                values = f[key]["values"][:]
                columns = list(f[key].attrs["columns"])
                data[key] = nap.TsdFrame(t=timestamps, d=values, columns=columns, time_units="s")
            elif obj_type == "pynapple_intervalset":
                start_times = f[key]["start_times"][:]
                end_times = f[key]["end_times"][:]
                data[key] = nap.IntervalSet(start=start_times, end=end_times, time_units="s")
            else:
                raise ValueError(f"Unknown type '{obj_type}' for key '{key}'")

    print(f"Loaded data from {full_path}")
    return data


