"""Batch SpyGlass insertion script for the Moore 2025 dataset.

This module provides functionality for inserting all converted NWB files from the
Moore 2025 dataset into a SpyGlass database. It handles batch processing, database
cleanup, and progress tracking for large-scale database population operations.
"""

import datajoint as dj
from pathlib import Path
import sys
from tqdm import tqdm

dj_local_conf_path = "/Users/pauladkisson/Documents/CatalystNeuro/Spyglass/spyglass/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# General Spyglass Imports
import spyglass.common as sgc  # this import connects to the database

# Custom Table Imports
sys.path.append(
    "/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/spyglass_extensions"
)
from imported_pseudo_emg import ImportedPseudoEMG
from imported_histology_images import ImportedHistologyImages

from insert_session import insert_session


def main():
    """Insert all Moore 2025 NWB files into SpyGlass database.

    Performs batch insertion of all converted NWB files for the Moore 2025 dataset.
    The function clears existing database entries, discovers all NWB files for the
    dataset subjects, and inserts them with progress tracking.

    The function suppresses logging and warnings for cleaner progress display.
    """
    # Suppress logging and warnings for cleaner progress bar
    import logging
    import warnings

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    (sgc.ProbeType & {"probe_type": "Cambridge Neurotech H5 probe"}).delete()
    (sgc.ProbeType & {"probe_type": "Cambridge Neurotech H6b probe"}).delete()
    (sgc.ProbeType & {"probe_type": "Cambridge Neurotech H7 probe"}).delete()
    (sgc.DataAcquisitionDevice & {"name": "data_acquisition_device"}).delete()
    (sgc.CameraDevice & {"camera_name": "Basler Camera"}).delete()
    (sgc.CameraDevice & {"camera_name": "Logitech Camera"}).delete()
    sgc.Task().delete()
    sgc.Nwbfile.delete()

    spyglass_raw_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw")
    nwbfile_paths = sorted(spyglass_raw_path.glob("*.nwb"))
    for nwbfile_path in tqdm(nwbfile_paths, desc="Inserting sessions"):
        # TODO: unskip these files after investigating timestamp issues and implementing fixes in conversion code
        if nwbfile_path.name == "H3019-210618_1.nwb":
            continue # skip this file due to known issues with SpatialSeries timestamps
        insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)


if __name__ == "__main__":
    main()
    print("Done!")
