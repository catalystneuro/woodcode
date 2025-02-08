

import os
from pathlib import Path
from typing import List
import pynapple as nap

import os
from pathlib import Path
from typing import List
import pynapple as nap


def create_nwb_file_list(data_dir: str, output_file: str, recursive: bool = False) -> int:
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

    Raises
    ------
    FileNotFoundError
        If data_dir doesn't exist
    ValueError
        If no NWB files are found
    """
    data_dir = Path(data_dir).resolve()  # Get absolute path
    output_file = Path(output_file)

    # Check if directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Find all .nwb files
    if recursive:
        nwb_files = list(data_dir.rglob("*.nwb"))
    else:
        nwb_files = list(data_dir.glob("*.nwb"))

    if not nwb_files:
        raise ValueError(f"No NWB files found in {data_dir}")

    # Convert to absolute paths
    absolute_paths = [str(f.resolve()) for f in nwb_files]

    # Sort alphabetically for consistency
    absolute_paths.sort()

    # Write to .list file
    with open(output_file, 'w') as f:
        for file_path in absolute_paths:
            f.write(f"{file_path}\n")

    return len(absolute_paths)


def load_nwb_files(file_list_path: str) -> List[nap.NWBFile]:
    """
    Load multiple NWB files using Pynapple from a list file containing full paths.

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
        If file_list_path is empty or contains invalid file names
    """
    file_list_path = Path(file_list_path)

    # Check if file list exists
    if not file_list_path.exists():
        raise FileNotFoundError(f"File list not found: {file_list_path}")

    # Read file paths from the list
    with open(file_list_path, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]

    if not file_paths:
        raise ValueError(f"No file paths found in {file_list_path}")

    # List to store loaded NWB files
    nwb_files = []

    # Load each NWB file
    for file_path in file_paths:
        path = Path(file_path)

        # Ensure file exists and has .nwb extension
        if not path.exists():
            raise FileNotFoundError(f"NWB file not found: {path}")
        if path.suffix.lower() != '.nwb':
            raise ValueError(f"File is not an NWB file: {path}")

        try:
            # Load the NWB file using Pynapple
            nwb_file = nap.load_file(str(path))
            nwb_files.append(nwb_file)
        except Exception as e:
            raise ValueError(f"Error loading NWB file {path}: {str(e)}")

    return nwb_files



