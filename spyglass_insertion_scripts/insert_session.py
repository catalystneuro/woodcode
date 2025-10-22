"""SpyGlass database insertion script for the example dataset.

This module provides functions for inserting converted NWB files from the example dataset into a SpyGlass database. 
It handles data ingestion, custom table population, and validation testing for the database integration pipeline.
"""

from pynwb import NWBHDF5IO
import numpy as np
import datajoint as dj
from pathlib import Path
import sys

dj_local_conf_path = "/Users/pauladkisson/Documents/CatalystNeuro/Spyglass/spyglass/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# General Spyglass Imports
import spyglass.common as sgc  # this import connects to the database
import spyglass.data_import as sgi
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
import spyglass.lfp as sglfp


def insert_session(nwbfile_path: Path, rollback_on_fail: bool = True, raise_err: bool = False):
    """Insert complete session data from NWB file into SpyGlass database.

    Performs comprehensive insertion of all session data including standard
    SpyGlass tables and custom task-specific tables. This is the main entry
    point for database ingestion of converted NWB files.

    Parameters
    ----------
    nwbfile_path : Path
        Path to the converted NWB file to insert into the database.
    rollback_on_fail : bool, optional
        Whether to rollback database transaction if insertion fails, by default True.
    raise_err : bool, optional
        Whether to raise exceptions on insertion errors, by default False.
    """
    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=rollback_on_fail, raise_err=raise_err)


def print_tables(nwbfile_path: Path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    with open("tables.txt", "w") as f:
        # NWB file and Subject info
        print("=== NWB File ===", file=f)
        print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Session ===", file=f)
        print(sgc.Session & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Subject ===", file=f)
        print(sgc.Subject(), file=f)

        # Task/Epoch tables
        print("=== IntervalList ===", file=f)
        print(sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Task ===", file=f)
        print(sgc.Task(), file=f)
        print("=== Task Epoch ===", file=f)
        print(sgc.TaskEpoch & {"nwb_file_name": nwb_copy_file_name}, file=f)

        # Video and Camera tables
        print("=== Video File ===", file=f)
        print(sgc.VideoFile & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Camera Device ===", file=f)
        print(sgc.CameraDevice(), file=f)

        # Electrode/Probe tables
        print("=== Data Acquisition Device ===", file=f)
        print(sgc.DataAcquisitionDevice(), file=f)
        print("=== Electrode ===", file=f)
        print(sgc.Electrode & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Electrode Group ===", file=f)
        print(sgc.ElectrodeGroup & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Probe ===", file=f)
        print(sgc.Probe & {"probe_id": "Cambridge Neurotech H7 probe"}, file=f)
        print("=== Probe Shank ===", file=f)
        print(sgc.Probe.Shank & {"probe_id": "Cambridge Neurotech H7 probe"}, file=f)
        print("=== Probe Electrode ===", file=f)
        print(sgc.Probe.Electrode & {"probe_id": "Cambridge Neurotech H7 probe"}, file=f)
        print("=== Raw ===", file=f)
        print(sgc.Raw & {"nwb_file_name": nwb_copy_file_name}, file=f)

        # LFP tables
        print("=== ImportedLFP ===", file=f)
        print(sglfp.ImportedLFP & {"nwb_file_name": nwb_copy_file_name}, file=f)

        # Tracking tables
        print("=== PositionSource ===", file=f)
        print(sgc.PositionSource & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== PositionSource.SpatialSeries ===", file=f)
        print(sgc.PositionSource.SpatialSeries & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== RawPosition.PosObject ===", file=f)
        print(sgc.RawPosition.PosObject & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== RawPosition ===", file=f)
        print((sgc.RawPosition & {"nwb_file_name": nwb_copy_file_name}).fetch1_dataframe(), file=f)


def main():
    nwbfile_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw/H7115-250618.nwb")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)

    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.ProbeType & {"probe_type": "Cambridge Neurotech H7 probe"}).delete()
    (sgc.DataAcquisitionDevice & {"name": "data_acquisition_device"}).delete()
    sgc.Task.delete()

    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path)


if __name__ == "__main__":
    main()
    print("Done!")
