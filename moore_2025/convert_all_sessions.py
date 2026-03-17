"""Primary script to run to convert all sessions in the Moore 2025 dataset."""
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
STREAM_NAME_PER_SESSION: dict[str, str] = {
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
    "H3015-210416_1": "Rhythm_FPGA-100.0",
    "H3015-210416_2": "Rhythm_FPGA-100.0",
    "H3015-210417": "Rhythm_FPGA-100.0",
    "H3016-210422": "Rhythm_FPGA-100.0",
    "H3016-210423": "Rhythm_FPGA-100.0",
    "H3019-210617": "Rhythm_FPGA-100.0",
    "H3019-210618_1": "Rhythm_FPGA-100.0",
    "H3022-210805": "Rhythm_FPGA-100.0",
    "H3022-210806": "Rhythm_FPGA-100.0",
    "H3023-210812": "Rhythm_FPGA-100.0",  # no raw ephys; entry unused
    "H3023-210813_1": "Rhythm_FPGA-100.0",
    "H3026-211003": "Rhythm_FPGA-100.0",
    "H3026-211004_2": "Rhythm_FPGA-100.0",
    "H3029-230510": "Acquisition_Board-100.Rhythm Data",  # exception
    # Adult sessions (default stream: "Rhythm_FPGA-103.0")
    "H4813-220728": "Rhythm_FPGA-103.0",
    "H4815-220814": "Rhythm_FPGA-103.0",
    "H4817-220828": "Rhythm_FPGA-103.0",
    "H4819-220929": "Rhythm_FPGA-103.0",
    "H4820-221007": "Rhythm_FPGA-103.0",
    "H4822-221023": "Rhythm_FPGA-103.0",
    "H4823-221108": "Rhythm_FPGA-103.0",
    "H4824-221117": "Rhythm_FPGA-103.0",
    "H4825-221124": "Rhythm_FPGA-103.0",
    "H4826-221203": "Rhythm_FPGA-103.0",
    "H4827-221210": "Rhythm_FPGA-103.0",
    "H4830-230406": "Record Node 103#Acquisition_Board-100.Rhythm Data",  # exception
}

# Whether each session has raw Open Ephys data. False means only processed data
# (e.g. .dat / .lfp) is available; get_session_to_nwb_kwargs uses this to switch
# to the no-raw-data path.
HAS_RAW_DATA_PER_SESSION: dict[str, bool] = {
    # Juvenile sessions
    "H3001-200201": True,
    "H3001-200202": True,
    "H3003-200207": True,
    "H3003-200208": True,
    "H3006-200314_1": True,
    "H3006-200314_2": True,
    "H3008-200805": True,
    "H3008-200807": True,
    "H3009-200812": True,
    "H3009-200813": True,
    "H3015-210416_1": True,
    "H3015-210416_2": True,
    "H3015-210417": True,
    "H3016-210422": True,
    "H3016-210423": True,
    "H3019-210617": True,
    "H3019-210618_1": True,
    "H3022-210805": True,
    "H3022-210806": True,
    "H3023-210812": False,   # Raw data missing
    "H3023-210813_1": False,  # Raw data missing
    "H3026-211003": False,   # Raw data missing
    "H3026-211004_2": False,  # Raw data missing
    "H3029-230510": True,
    # Adult sessions
    "H4813-220728": True,
    "H4815-220814": True,
    "H4817-220828": True,
    "H4819-220929": True,
    "H4820-221007": True,
    "H4822-221023": True,
    "H4823-221108": False,   # Raw data and video missing
    "H4824-221117": True,
    "H4825-221124": True,
    "H4826-221203": True,
    "H4827-221210": True,
    "H4830-230406": True,
}

# Whether each session has video data recorded.
HAS_VIDEO_PER_SESSION: dict[str, bool] = {
    # Juvenile sessions
    "H3001-200201": False,   # Video not recorded
    "H3001-200202": False,   # Video not recorded
    "H3003-200207": True,
    "H3003-200208": True,
    "H3006-200314_1": True,
    "H3006-200314_2": True,
    "H3008-200805": True,
    "H3008-200807": True,
    "H3009-200812": True,
    "H3009-200813": True,
    "H3015-210416_1": True,
    "H3015-210416_2": True,
    "H3015-210417": True,
    "H3016-210422": True,
    "H3016-210423": True,
    "H3019-210617": True,
    "H3019-210618_1": True,
    "H3022-210805": True,
    "H3022-210806": True,
    "H3023-210812": True,
    "H3023-210813_1": True,
    "H3026-211003": True,
    "H3026-211004_2": True,
    "H3029-230510": True,
    # Adult sessions
    "H4813-220728": True,
    "H4815-220814": True,
    "H4817-220828": True,
    "H4819-220929": True,
    "H4820-221007": True,
    "H4822-221023": False,   # Video missing
    "H4823-221108": False,   # Raw data and video missing
    "H4824-221117": True,
    "H4825-221124": True,
    "H4826-221203": True,
    "H4827-221210": True,
    "H4830-230406": True,
}


def detect_raw_ephys_paths(raw_folder_path: Path) -> tuple[Path, Path]:
    """Find the Open Ephys root folder and continuous.xml path within a Raw directory.

    The Open Ephys root folder is defined as the folder containing settings.xml.

    Parameters
    ----------
    raw_folder_path : Path
        The session's Raw directory (or any folder containing Open Ephys output).

    Returns
    -------
    tuple[Path, Path]
        (raw_ephys_folder_path, continuous_xml_path)
    """
    raw_ephys_folder_path = next(raw_folder_path.rglob("experiment*")).parent
    continuous_xml_path = next(raw_folder_path.rglob("continuous.xml"))
    return raw_ephys_folder_path, continuous_xml_path


def detect_processed_paths(
    session_folder_path: Path, folder_name: str
) -> tuple[Path, Path, Path, Path, Path]:
    """Detect processed file paths, handling flat and nested Processed directory layouts.

    Parameters
    ----------
    session_folder_path : Path
        Path to the session folder (e.g. WT/H3022-210805/).
    folder_name : str
        Session folder name (e.g. "H3022-210805").

    Returns
    -------
    tuple[Path, Path, Path, Path, Path]
        (processed_xml_path, nrs_path, lfp_file_path, mat_path, sleep_path)
    """
    processed_root = session_folder_path / "Processed"
    nested_directory = processed_root / folder_name

    if nested_directory.is_dir():
        base = nested_directory
    else:
        base = processed_root

    processed_xml_path = base / f"{folder_name}.xml"
    nrs_path = base / f"{folder_name}.nrs"
    lfp_file_path = base / f"{folder_name}.lfp"
    mat_path = base / "Analysis"
    sleep_path = base / "Sleep"
    return processed_xml_path, nrs_path, lfp_file_path, mat_path, sleep_path


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
    processed_xml_path, nrs_path, lfp_file_path, mat_path, sleep_path = detect_processed_paths(
        session_folder_path, folder_name
    )

    has_raw_data = HAS_RAW_DATA_PER_SESSION[folder_name]
    has_video = HAS_VIDEO_PER_SESSION[folder_name]

    if has_raw_data:
        raw_ephys_folder_path, raw_xml_path = detect_raw_ephys_paths(raw_folder_path)
        stream_name = STREAM_NAME_PER_SESSION[folder_name]
        ttl_stream_name = f"{stream_name}_ADC"

        if folder_name == "H4817-220828":
            raw_xml_path = processed_xml_path  # raw XML is missing a channel

        video_file_paths, timestamps_file_paths = detect_video_and_timestamp_paths(raw_folder_path)
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
            video_file_paths=video_file_paths,
            is_adult=is_adult,
        )
    else:
        # No raw Open Ephys data; use .dat file from Processed/.
        processed_root = session_folder_path / "Processed"
        raw_xml_path = processed_xml_path
        raw_ephys_folder_path = None
        raw_ephys_dat_file_path = processed_root / f"{folder_name}.dat"
        stream_name = None
        ttl_stream_name = None

        video_file_paths, timestamps_file_paths = detect_video_and_timestamp_paths(processed_root)
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
            raw_ephys_folder_path=raw_ephys_folder_path,
            raw_ephys_dat_file_path=raw_ephys_dat_file_path,
            video_file_paths=video_file_paths,
            is_adult=is_adult,
        )


def collect_session_to_nwb_kwargs_per_session(
    *,
    data_dir_path: Path,
    meta_path: Path,
    histology_folder_path: Path,
    juvenile_metadata_file_path: Path,
    adult_metadata_file_path: Path,
) -> list[dict[str, Any]]:
    """Collect the kwargs for session_to_nwb for each session in the dataset.

    Parameters
    ----------
    data_dir_path : Path
        Root directory containing H3000_Juveniles/ and H4800_Adults/.
    meta_path : Path
        Path to the shared Excel metadata file (MooreDataset_Metadata.xlsx).
    histology_folder_path : Path
        Top-level histology folder (contains H3000/ and H4800/ subdirectories).
    juvenile_metadata_file_path : Path
        Path to juvenile_metadata.yaml.
    adult_metadata_file_path : Path
        Path to adult_metadata.yaml.

    Returns
    -------
    list[dict[str, Any]]
        A list of kwargs dicts for session_to_nwb, one per session.
    """
    all_session_to_nwb_kwargs = []
    cohort_folders = [("H3000_Juveniles", False), ("H4800_Adults", True)]
    for cohort_folder_name, is_adult in cohort_folders:
        cohort_folder_path = data_dir_path / cohort_folder_name
        for genotype in ["WT", "KO"]:
            genotype_folder_path = cohort_folder_path / genotype
            for session_folder_path in sorted(genotype_folder_path.iterdir()):
                if session_folder_path.name.startswith("."):
                    continue
                if not session_folder_path.is_dir():
                    continue
                folder_name = session_folder_path.name
                metadata_file_path = adult_metadata_file_path if is_adult else juvenile_metadata_file_path
                histology_subfolder = histology_folder_path / ("H4800" if is_adult else "H3000")
                session_to_nwb_kwargs = get_session_to_nwb_kwargs(
                    session_folder_path=session_folder_path,
                    folder_name=folder_name,
                    is_adult=is_adult,
                    meta_path=meta_path,
                    metadata_file_path=metadata_file_path,
                    histology_folder_path=histology_subfolder,
                )
                all_session_to_nwb_kwargs.append(session_to_nwb_kwargs)
    return all_session_to_nwb_kwargs


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
    try:
        session_to_nwb(**session_to_nwb_kwargs)
    except Exception as error:
        with open(exception_file_path, mode="w") as file:
            file.write(f"session_to_nwb_kwargs: \n {pformat(session_to_nwb_kwargs)}\n\n")
            file.write(traceback.format_exc())


def dataset_to_nwb(
    *,
    data_dir_path: Path,
    output_dir_path: Path,
    meta_path: Path,
    histology_folder_path: Path,
    juvenile_metadata_file_path: Path,
    adult_metadata_file_path: Path,
    stub_test: bool = False,
    max_workers: int = 1,
):
    """Convert all sessions in the Moore 2025 dataset to NWB.

    Parameters
    ----------
    data_dir_path : Path
        Root directory containing H3000_Juveniles/ and H4800_Adults/.
    output_dir_path : Path
        Directory where NWB files will be saved.
    meta_path : Path
        Path to the shared Excel metadata file (MooreDataset_Metadata.xlsx).
    histology_folder_path : Path
        Top-level histology folder (contains H3000/ and H4800/ subdirectories).
    juvenile_metadata_file_path : Path
        Path to juvenile_metadata.yaml.
    adult_metadata_file_path : Path
        Path to adult_metadata.yaml.
    stub_test : bool, optional
        Whether to stub data for testing, by default False.
    max_workers : int, optional
        The number of workers to use for parallel processing, by default 1.
    """
    session_to_nwb_kwargs_per_session = collect_session_to_nwb_kwargs_per_session(
        data_dir_path=data_dir_path,
        meta_path=meta_path,
        histology_folder_path=histology_folder_path,
        juvenile_metadata_file_path=juvenile_metadata_file_path,
        adult_metadata_file_path=adult_metadata_file_path,
    )
    print(f"Collected session_to_nwb kwargs for {len(session_to_nwb_kwargs_per_session)} sessions.")
    return

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
    data_dir_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset")
    meta_path = data_dir_path / "MooreDataset_Metadata.xlsx"
    histology_folder_path = data_dir_path / "Histology"
    output_dir_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw")
    juvenile_metadata_file_path = Path(__file__).parent / "juvenile_metadata.yaml"
    adult_metadata_file_path = Path(__file__).parent / "adult_metadata.yaml"
    stub_test = False
    max_workers = 10

    dataset_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        meta_path=meta_path,
        histology_folder_path=histology_folder_path,
        juvenile_metadata_file_path=juvenile_metadata_file_path,
        adult_metadata_file_path=adult_metadata_file_path,
        stub_test=stub_test,
        max_workers=max_workers,
    )
