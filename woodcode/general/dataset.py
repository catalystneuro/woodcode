import os
from pathlib import Path
from typing import List
import warnings
import pynapple as nap
import pandas as pd
import numpy as np

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


def load_nwb_files(file_list_path: str) -> List[nap.NWBFile]:
    """
    Load multiple NWB files using Pynapple from a list file containing full paths.
    Lines starting with # are treated as comments and ignored.

    Parameters
    ----------
    file_list_path : str
        Path to the text file containing full paths to NWB files (one per line)

    Returns
    -------
    List[nap.NWBFile]
        List of loaded Pynapple NWB file objects

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
