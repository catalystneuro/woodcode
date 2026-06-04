"""Reusable Spyglass insertion logic for the Moore 2025 dataset.

Defines ``insert_session()``, which ingests a single converted NWB file into a Spyglass database, along
with its custom-insert helpers (``insert_sleep``, ``insert_sorting``, ``insert_pseudo_emg``,
``insert_histology_images``), ``print_tables()`` for QA dumps, and ``clear_shared_tables()`` for clearing
the shared probe/camera/device/task records before a fresh insertion. The driver scripts
(``insert_single_session.py``, ``insert_example_sessions.py``, ``insert_edge_case_sessions.py``) and
``insert_all_sessions.py`` import from here.

Importing this module loads ``dj_local_conf.json`` and connects to the database.
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

# Spike Sorting Imports
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
import spyglass.spikesorting.v1 as sgs
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams
from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation

# Custom Table Imports
sys.path.append(
    "/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/spyglass_extensions"
)
from imported_pseudo_emg import ImportedPseudoEMG
from imported_histology_images import ImportedHistologyImages

def insert_pseudo_emg(nwbfile_path: Path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    ImportedPseudoEMG().make(key={"nwb_file_name": nwb_copy_file_name})

def insert_histology_images(nwbfile_path: Path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    ImportedHistologyImages().make(key={"nwb_file_name": nwb_copy_file_name})

def insert_sorting(nwbfile_path: Path):
    """Insert spike sorting data and unit annotations into SpyGlass database.

    Creates a sorted spikes group containing all units from the imported spike
    sorting data and adds annotations for each unit. The annotations include
    sampling rate and mean waveform data extracted from the NWB file.

    Parameters
    ----------
    nwbfile_path : Path
        Path to the NWB file containing spike sorting results and unit metadata.

    Notes
    -----
    The function creates a group named "all_units" and adds the following
    annotations for each unit:
    - sampling_rate: The sampling rate for the unit
    - waveform_mean: The mean waveform for the unit
    """
    io = NWBHDF5IO(nwbfile_path, "r")
    nwbfile = io.read()
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    merge_id = str((SpikeSortingOutput.ImportedSpikeSorting & {"nwb_file_name": nwb_copy_file_name}).fetch1("merge_id"))

    UnitSelectionParams().insert_default()
    group_name = "all_units"
    SortedSpikesGroup().create_group(
        group_name=group_name,
        nwb_file_name=nwb_copy_file_name,
        keys=[{"spikesorting_merge_id": merge_id}],
    )
    group_key = {
        "nwb_file_name": nwb_copy_file_name,
        "sorted_spikes_group_name": group_name,
    }
    group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
    _, unit_ids = SortedSpikesGroup().fetch_spike_data(group_key, return_unit_ids=True)

    for unit_key in unit_ids:
        unit_id = unit_key["unit_id"]
        sampling_rate = nwbfile.units.get((unit_id, "sampling_rate"))
        waveform_mean = nwbfile.units.get((unit_id, "waveform_mean"))
        annotations = {
            "sampling_rate": sampling_rate,
            "waveform_mean": waveform_mean,
        }
        sgs.ImportedSpikeSorting().add_annotation(key={"nwb_file_name": nwb_copy_file_name}, id=unit_id, annotations=annotations)
    io.close()

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
    insert_sleep(nwbfile_path)
    insert_sorting(nwbfile_path)
    insert_pseudo_emg(nwbfile_path)
    insert_histology_images(nwbfile_path)

def insert_sleep(nwbfile_path: Path):
    nwb_copy_filename = get_nwb_copy_filename(nwbfile_path.name)
    with NWBHDF5IO(str(nwbfile_path), "r") as io:
        nwbfile = io.read()
        sleep_stages = nwbfile.intervals["sleep_stages"].to_dataframe()
    unique_tags = ["rem", "nrem", "wake"]
    for tag in unique_tags:
        stage_intervals = sleep_stages[sleep_stages["tags"] == tag]
        start_times = stage_intervals["start_time"].to_numpy()
        stop_times = stage_intervals["stop_time"].to_numpy()
        valid_times = np.column_stack((start_times, stop_times))
        key = {"nwb_file_name": nwb_copy_filename, "interval_list_name": f"sleep_{tag}", "valid_times": valid_times}
        sgc.IntervalList().insert1(key)


def clear_shared_tables():
    """Delete the shared probe/camera/device/task records before a fresh insertion.

    These records are shared across sessions, so they are cleared once at the start of an insertion run
    (and re-created by ``insert_session``). Deleting ``ProbeType`` cascades to everything referencing it,
    so this resets any previously inserted sessions as well.
    """
    (sgc.ProbeType & {"probe_type": "Cambridge Neurotech H5 probe"}).delete()
    (sgc.ProbeType & {"probe_type": "Cambridge Neurotech H6b probe"}).delete()
    (sgc.ProbeType & {"probe_type": "Cambridge Neurotech H7 probe"}).delete()
    (sgc.DataAcquisitionDevice & {"name": "data_acquisition_device"}).delete()
    (sgc.CameraDevice & {"camera_name": "Basler Camera"}).delete()
    (sgc.CameraDevice & {"camera_name": "Logitech Camera"}).delete()
    sgc.Task().delete()


def print_tables(nwbfile_path: Path, table_path: Path = Path("tables.txt")):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    with open(table_path, "w") as f:
        # NWB file and Subject info
        print("=== NWB File ===", file=f)
        print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Session ===", file=f)
        print(sgc.Session & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Subject ===", file=f)
        print(sgc.Subject(), file=f)

        # Task/Epoch/Sleep tables
        print("=== IntervalList ===", file=f)
        print(sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Task ===", file=f)
        print(sgc.Task(), file=f)
        print("=== Task Epoch ===", file=f)
        print(sgc.TaskEpoch & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Sleep NREM Valid Times ===", file=f)
        print((sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name, "interval_list_name": "sleep_nrem"}).fetch1("valid_times"), file=f)

        # PseudoEMG table
        print("=== ImportedPseudoEMG ===", file=f)
        print(ImportedPseudoEMG & {"nwb_file_name": nwb_copy_file_name}, file=f)

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
        print((sgc.RawPosition & {"nwb_file_name": nwb_copy_file_name, "interval_list_name": "pos 0 valid times"}).fetch1_dataframe(), file=f)
        print("=== RawCompassDirection ===", file=f)
        print((sgc.RawCompassDirection & {"nwb_file_name": nwb_copy_file_name}), file=f)

        # Spike Sorting tables
        print("=== ImportedSpikeSorting ===", file=f)
        print(sgs.ImportedSpikeSorting & {"nwb_file_name": nwb_copy_file_name}, file=f)
        merge_id = str((SpikeSortingOutput.ImportedSpikeSorting & {"nwb_file_name": nwb_copy_file_name}).fetch1("merge_id"))
        print("=== Annotation ===", file=f)
        print(sgs.ImportedSpikeSorting.Annotations & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Example Annotation (Unit 0) ===", file=f)
        print((sgs.ImportedSpikeSorting.Annotations & {"nwb_file_name": nwb_copy_file_name, "id": 0}).fetch1("annotations"), file=f)

        # Custom Histology Images table
        print("=== ImportedHistologyImages ===", file=f)
        print(ImportedHistologyImages & {"nwb_file_name": nwb_copy_file_name}, file=f)
