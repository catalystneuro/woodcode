"""Primary script to run to convert all sessions in the Moore 2025 dataset."""
import contextlib
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pprint import pformat
from typing import Any, Union

from tqdm import tqdm

from moore_2025.convert_session import session_to_nwb

# Manually specified stream names for each session.
# The ttl_stream_name is always derived as f"{stream_name}_ADC".
# Sessions that use a .dat file instead of Open Ephys output are handled as edge cases
# in get_session_to_nwb_kwargs and their entries here are unused.
STREAM_NAME_PER_SESSION: dict[str, str | None] = {
    # Juvenile sessions (default stream: "Rhythm_FPGA-100.0")
    "H3001-200201": "Rhythm_FPGA-100.0",
    "H3001-200202": "Rhythm_FPGA-100.0",
    "H3003-200207": "Rhythm_FPGA-100.0",
    "H3003-200208": "Rhythm_FPGA-100.0",
    "H3006-200314_1": "Rhythm_FPGA-100.0",
    "H3006-200314_2": "Rhythm_FPGA-100.0",
    "H3008-200805": "Rhythm_FPGA-100.0",
    "H3008-200807": "Rhythm_FPGA-100.0",
    "H3009-200812": "Rhythm_FPGA-100.0",
    "H3009-200813": "Rhythm_FPGA-100.0",
    "H3015-210416_1": None,
    "H3015-210416_2": "Rhythm_FPGA-100.0",
    "H3015-210417": "Rhythm_FPGA-100.0",
    "H3016-210422": "Rhythm_FPGA-100.0",
    "H3016-210423": "Rhythm_FPGA-100.0",
    "H3019-210617": "Rhythm_FPGA-100.0",
    "H3019-210618_1": "Rhythm_FPGA-100.0",
    "H3022-210805": "Rhythm_FPGA-100.0",
    "H3022-210806": "Rhythm_FPGA-100.0",
    "H3023-210812": None,    # raw data missing
    "H3023-210813_1": None,  # raw data missing
    "H3026-211003": None,    # raw data missing
    "H3026-211004_2": None,  # raw data missing
    "H3029-230510": "Acquisition_Board-100.Rhythm Data",  # exception
    # Adult sessions (default stream: "Rhythm_FPGA-103.0")
    "H4813-220728": "Rhythm_FPGA-103.0",
    "H4815-220814": "Rhythm_FPGA-103.0",
    "H4817-220828": "Rhythm_FPGA-103.0",
    "H4819-220929": "Rhythm_FPGA-103.0",
    "H4820-221007": "Rhythm_FPGA-103.0",
    "H4822-221023": "Rhythm_FPGA-103.0",
    "H4823-221108": None,    # raw data and video missing
    "H4824-221117": "Rhythm_FPGA-103.0",
    "H4825-221124": "Record Node 101#Acquisition_Board-100.Rhythm Data",
    "H4826-221203": "Record Node 102#Acquisition_Board-101.Rhythm Data",
    "H4827-221210": "Record Node 101#Acquisition_Board-100.Rhythm Data",
    "H4830-230406": "Record Node 103#Acquisition_Board-100.Rhythm Data",  # exception
}

# Sessions missing raw Open Ephys data; only processed data (.dat / .lfp) is available.
SESSIONS_WITHOUT_RAW_DATA: set[str] = {
    "H3023-210812",    # Raw data missing
    "H3023-210813_1",  # Raw data missing
    "H3026-211003",    # Raw data missing
    "H3026-211004_2",  # Raw data missing
    "H3015-210416_1",  # Raw data missing
    "H4823-221108",    # Raw data and video missing
}

# Sessions with no video recorded.
SESSIONS_WITHOUT_VIDEO: set[str] = {
    "H3001-200201",  # Video not recorded
    "H3001-200202",  # Video not recorded
    "H4822-221023",  # Video missing
    "H4823-221108",  # Raw data and video missing
}

# Sessions that should use Processed/<session>.xml for raw_xml_path, even if raw data exists.
SESSIONS_USING_PROCESSED_XML: set[str] = {
    "H3029-230510", # Raw XML for this session is missing the SpikeGroup section, so using the Processed XML instead
    "H4817-220828", # Raw XML for this session is missing one of the channels (channel 38 on shank 1), so using the Processed XML instead
}

# Sessions to skip entirely (not converted).
SESSIONS_TO_SKIP: set[str] = {
    "H3009-200813",      # Missing .nrs file
    "H3015-210416_2",    # Missing .nrs file
    "H3015-210417",      # Missing .nrs file
    "H3008-200807",      # Missing .nrs file
    "H3019-210617",      # Missing .nrs file
    "H3019-210618_1",    # Missing .nrs file
    "H3015-210416_1",    # Missing .nrs file
    "H3023-210813_1",    # Missing .nrs file
    "H3006-200314_1",    # Multi-experiment session
    "H3026-211003",      # Mismatched metadata and XML (metadata specifies 32 channels but XML contains 64)
    "H3026-211004_2",    # Mismatched metadata and XML (metadata specifies 32 channels but XML contains 64)
}

# TODO: Consider a more robust solution.
SESSION_TO_ALT_XML_FOLDER_PATH: dict[str, Path] = {
    "H4815-220814": Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/H4800_Adults/WT/H4820-221007"), # XMLs for this session are missing a channel, so using a neighbor instead
}


def detect_video_and_timestamp_paths(raw_folder_path: Path) -> tuple[list[Path] | None, list[Path]]:
    """Detect video and timestamp files in a Raw directory.

    Parameters
    ----------
    raw_folder_path : Path
        The session's Raw directory (or subfolder containing Bonsai files).

    Returns
    -------
    tuple[list[Path] | None, list[Path]]
        (video_file_paths, timestamps_file_paths). video_file_paths is None if no videos found.
    """
    video_file_paths = sorted(raw_folder_path.rglob("Bonsai*.avi"))
    timestamps_file_paths = sorted(raw_folder_path.rglob("Bonsai*.csv"))
    if not video_file_paths:
        video_file_paths = None
    return video_file_paths, timestamps_file_paths


def get_session_to_nwb_kwargs(
    session_folder_path: Path,
    folder_name: str,
    is_adult: bool,
    meta_path: Path,
    metadata_file_path: Path,
    histology_folder_path: Path,
) -> dict[str, Any]:
    """Build the kwargs dict for session_to_nwb for a single session.

    Stream names are looked up from STREAM_NAME_PER_SESSION. File paths are
    detected from the directory structure. Named edge-case branches handle
    sessions with unusual directory structures.

    Parameters
    ----------
    session_folder_path : Path
        Path to the session folder (e.g. H3000_Juveniles/WT/H3022-210805/).
    folder_name : str
        Session folder name (e.g. "H3022-210805").
    is_adult : bool
        Whether the subject is an adult (True) or juvenile (False).
    meta_path : Path
        Path to the shared Excel metadata file.
    metadata_file_path : Path
        Path to the YAML metadata file (juvenile_metadata.yaml or adult_metadata.yaml).
    histology_folder_path : Path
        Path to the cohort-level histology folder (e.g. Histology/H3000).

    Returns
    -------
    dict[str, Any]
        Kwargs to pass to session_to_nwb (excluding save_path and stub_test).
    """
    print(f"Collecting session_to_nwb kwargs for session {folder_name}...")
    if folder_name == "H3029-230510": # uses adult-style temporal alignment
        is_adult = True
    raw_folder_path = session_folder_path / "Raw"
    processed_root = session_folder_path / "Processed"
    nested_directory = processed_root / folder_name
    processed_folder_path = nested_directory if nested_directory.is_dir() else processed_root
    processed_xml_path = processed_folder_path / f"{folder_name}.xml"
    try:
        nrs_path = next(processed_folder_path.glob("*.nrs"))
    except StopIteration:
        nrs_path = None
        print(f"Warning: No .nrs file found for session {folder_name} in {processed_folder_path}.")
    lfp_file_path = processed_folder_path / f"{folder_name}.lfp"
    mat_path = processed_folder_path / "Analysis"
    sleep_path = processed_folder_path / "Sleep"

    has_raw_data = folder_name not in SESSIONS_WITHOUT_RAW_DATA
    has_video = folder_name not in SESSIONS_WITHOUT_VIDEO

    # Defaults for optional kwargs; overridden below when applicable.
    raw_ephys_folder_path = None
    raw_ephys_dat_file_path = None
    stream_name = None
    ttl_stream_name = None

    if has_raw_data:
        raw_ephys_folder_path = next(raw_folder_path.rglob("experiment*")).parent
        if folder_name in SESSION_TO_ALT_XML_FOLDER_PATH:
            alt_xml_raw_folder_path = SESSION_TO_ALT_XML_FOLDER_PATH[folder_name] / "Raw"
            raw_xml_path = next(alt_xml_raw_folder_path.rglob("continuous.xml"))
        else:
            raw_xml_path = next(raw_folder_path.rglob("continuous.xml"))
        if folder_name in SESSIONS_USING_PROCESSED_XML:
            raw_xml_path = processed_xml_path
        stream_name = STREAM_NAME_PER_SESSION[folder_name]
        ttl_stream_name = f"{stream_name}_ADC"
        video_file_paths, timestamps_file_paths = detect_video_and_timestamp_paths(raw_folder_path)
    else:
        # No raw Open Ephys data; use .dat file from Processed/.
        raw_xml_path = processed_xml_path
        raw_ephys_dat_file_path = processed_folder_path / f"{folder_name}.dat"
        video_file_paths, timestamps_file_paths = detect_video_and_timestamp_paths(processed_folder_path)

    if not has_video:
        video_file_paths = None

    return dict(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        metadata_file_path=metadata_file_path,
        histology_folder_path=histology_folder_path,
        stream_name=stream_name,
        ttl_stream_name=ttl_stream_name,
        raw_ephys_folder_path=raw_ephys_folder_path,
        raw_ephys_dat_file_path=raw_ephys_dat_file_path,
        video_file_paths=video_file_paths,
        is_adult=is_adult,
    )


def collect_juvenile_session_to_nwb_kwargs(
    *,
    juvenile_dir_path: Path,
    meta_path: Path,
    histology_folder_path: Path,
    metadata_file_path: Path,
) -> list[dict[str, Any]]:
    """Collect the kwargs for session_to_nwb for each juvenile session.

    Parameters
    ----------
    juvenile_dir_path : Path
        Root directory containing WT/ and KO/ subdirectories for juvenile sessions.
    meta_path : Path
        Path to the shared Excel metadata file (MooreDataset_Metadata.xlsx).
    histology_folder_path : Path
        Histology folder for the juvenile cohort (e.g. Histology/H3000/).
    metadata_file_path : Path
        Path to juvenile_metadata.yaml.

    Returns
    -------
    list[dict[str, Any]]
        A list of kwargs dicts for session_to_nwb, one per session.
    """
    session_to_nwb_kwargs_list = []
    for genotype in ["WT", "KO"]:
        genotype_folder_path = juvenile_dir_path / genotype
        for session_folder_path in sorted(genotype_folder_path.iterdir()):
            if session_folder_path.name.startswith("."):
                continue
            if not session_folder_path.is_dir():
                continue
            folder_name = session_folder_path.name
            if folder_name in SESSIONS_TO_SKIP:
                continue
            session_to_nwb_kwargs_list.append(
                get_session_to_nwb_kwargs(
                    session_folder_path=session_folder_path,
                    folder_name=folder_name,
                    is_adult=False,
                    meta_path=meta_path,
                    metadata_file_path=metadata_file_path,
                    histology_folder_path=histology_folder_path,
                )
            )
    return session_to_nwb_kwargs_list


def collect_adult_session_to_nwb_kwargs(
    *,
    adult_dir_path: Path,
    meta_path: Path,
    histology_folder_path: Path,
    metadata_file_path: Path,
) -> list[dict[str, Any]]:
    """Collect the kwargs for session_to_nwb for each adult session.

    Parameters
    ----------
    adult_dir_path : Path
        Root directory containing WT/ and KO/ subdirectories for adult sessions.
    meta_path : Path
        Path to the shared Excel metadata file (MooreDataset_Metadata.xlsx).
    histology_folder_path : Path
        Histology folder for the adult cohort (e.g. Histology/H4800/).
    metadata_file_path : Path
        Path to adult_metadata.yaml.

    Returns
    -------
    list[dict[str, Any]]
        A list of kwargs dicts for session_to_nwb, one per session.
    """
    session_to_nwb_kwargs_list = []
    for genotype in ["WT", "KO"]:
        genotype_folder_path = adult_dir_path / genotype
        for session_folder_path in sorted(genotype_folder_path.iterdir()):
            if session_folder_path.name.startswith("."):
                continue
            if not session_folder_path.is_dir():
                continue
            folder_name = session_folder_path.name
            if folder_name in SESSIONS_TO_SKIP:
                continue
            session_to_nwb_kwargs_list.append(
                get_session_to_nwb_kwargs(
                    session_folder_path=session_folder_path,
                    folder_name=folder_name,
                    is_adult=True,
                    meta_path=meta_path,
                    metadata_file_path=metadata_file_path,
                    histology_folder_path=histology_folder_path,
                )
            )
    return session_to_nwb_kwargs_list


def safe_session_to_nwb(*, session_to_nwb_kwargs: dict, exception_file_path: Union[Path, str]):
    """Convert a session to NWB while handling any errors by recording error messages to the exception_file_path.

    Parameters
    ----------
    session_to_nwb_kwargs : dict
        The arguments for session_to_nwb.
    exception_file_path : Union[Path, str]
        The path to the file where the exception messages will be saved.
    """
    exception_file_path = Path(exception_file_path)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        try:
            session_to_nwb(**session_to_nwb_kwargs)
        except Exception:
            with open(exception_file_path, mode="w") as file:
                file.write(f"session_to_nwb_kwargs: \n {pformat(session_to_nwb_kwargs)}\n\n")
                file.write(traceback.format_exc())


def dataset_to_nwb(
    *,
    juvenile_dir_path: Path,
    adult_dir_path: Path,
    output_dir_path: Path,
    meta_path: Path,
    juvenile_histology_folder_path: Path,
    adult_histology_folder_path: Path,
    juvenile_metadata_file_path: Path,
    adult_metadata_file_path: Path,
    stub_test: bool = False,
    max_workers: int = 1,
):
    """Convert all sessions in the Moore 2025 dataset to NWB.

    Parameters
    ----------
    juvenile_dir_path : Path
        Root directory containing WT/ and KO/ subdirectories for juvenile sessions.
    adult_dir_path : Path
        Root directory containing WT/ and KO/ subdirectories for adult sessions.
    output_dir_path : Path
        Directory where NWB files will be saved.
    meta_path : Path
        Path to the shared Excel metadata file (MooreDataset_Metadata.xlsx).
    juvenile_histology_folder_path : Path
        Histology folder for the juvenile cohort (e.g. Histology/H3000/).
    adult_histology_folder_path : Path
        Histology folder for the adult cohort (e.g. Histology/H4800/).
    juvenile_metadata_file_path : Path
        Path to juvenile_metadata.yaml.
    adult_metadata_file_path : Path
        Path to adult_metadata.yaml.
    stub_test : bool, optional
        Whether to stub data for testing, by default False.
    max_workers : int, optional
        The number of workers to use for parallel processing, by default 1.
    """
    session_to_nwb_kwargs_per_session = collect_juvenile_session_to_nwb_kwargs(
        juvenile_dir_path=juvenile_dir_path,
        meta_path=meta_path,
        histology_folder_path=juvenile_histology_folder_path,
        metadata_file_path=juvenile_metadata_file_path,
    ) + collect_adult_session_to_nwb_kwargs(
        adult_dir_path=adult_dir_path,
        meta_path=meta_path,
        histology_folder_path=adult_histology_folder_path,
        metadata_file_path=adult_metadata_file_path,
    )
    print(f"Collected session_to_nwb kwargs for {len(session_to_nwb_kwargs_per_session)} sessions.")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for session_to_nwb_kwargs in session_to_nwb_kwargs_per_session:
            session_to_nwb_kwargs["save_path"] = output_dir_path
            session_to_nwb_kwargs["stub_test"] = stub_test
            folder_name = session_to_nwb_kwargs["folder_name"]
            exception_file_path = output_dir_path / f"ERROR_{folder_name}.txt"
            futures.append(
                executor.submit(
                    safe_session_to_nwb,
                    session_to_nwb_kwargs=session_to_nwb_kwargs,
                    exception_file_path=exception_file_path,
                )
            )
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass


if __name__ == "__main__":
    juvenile_dir_path = Path("/Volumes/SamsungSSD/CatalystNeuro/Dudchenko/251104_MooreDataset/H3000_Juveniles")
    adult_dir_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/H4800_Adults")
    meta_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/MooreDataset_Metadata.xlsx")
    juvenile_histology_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/Histology/H3000")
    adult_histology_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/Histology/H4800")
    output_dir_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw")
    juvenile_metadata_file_path = Path(__file__).parent / "juvenile_metadata.yaml"
    adult_metadata_file_path = Path(__file__).parent / "adult_metadata.yaml"
    stub_test = True
    max_workers = 10

    dataset_to_nwb(
        juvenile_dir_path=juvenile_dir_path,
        adult_dir_path=adult_dir_path,
        output_dir_path=output_dir_path,
        meta_path=meta_path,
        juvenile_histology_folder_path=juvenile_histology_folder_path,
        adult_histology_folder_path=adult_histology_folder_path,
        juvenile_metadata_file_path=juvenile_metadata_file_path,
        adult_metadata_file_path=adult_metadata_file_path,
        stub_test=stub_test,
        max_workers=max_workers,
    )
