"""Primary script to run to convert all sessions in the Moore 2025 dataset."""
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pprint import pformat
from typing import Any, Union

from tqdm import tqdm

from moore_2025.convert_session import session_to_nwb


def detect_open_ephys_stream(raw_folder_path: Path) -> tuple[Path, Path, str, str]:
    """Detect the Open Ephys stream name and ephys root folder from a Raw directory.

    Handles two recording software versions:
    - Older format: Raw/experiment{N}/recording{M}/continuous/{stream}/continuous.xml
    - Newer format: Raw/{timestamp}/Record Node {N}/experiment{N}/recording{M}/continuous/{stream}/continuous.xml

    Parameters
    ----------
    raw_folder_path : Path
        The session's Raw directory (or any folder containing Open Ephys output).

    Returns
    -------
    tuple[Path, Path, str, str]
        (raw_ephys_folder_path, continuous_xml_path, stream_name, ttl_stream_name)
    """
    continuous_xml_path = next(raw_folder_path.rglob("continuous.xml"))

    # Navigate up from continuous.xml:
    # parents[0] = stream_folder (e.g. "Rhythm_FPGA-100.0")
    # parents[1] = "continuous"
    # parents[2] = recording_folder (e.g. "recording1")
    # parents[3] = experiment_folder (e.g. "experiment1") OR "Record Node 103"
    # parents[4] = potential root OR parent of "Record Node"
    stream_folder = continuous_xml_path.parents[0]
    potential_root = continuous_xml_path.parents[4]

    if potential_root.name.startswith("Record Node"):
        raw_ephys_folder_path = potential_root.parent
        stream_name = f"{potential_root.name}#{stream_folder.name}"
    else:
        raw_ephys_folder_path = potential_root
        stream_name = stream_folder.name

    ttl_stream_name = f"{stream_name}_ADC"
    return raw_ephys_folder_path, continuous_xml_path, stream_name, ttl_stream_name


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


def detect_video_and_timestamp_paths(
    raw_folder_path: Path, is_adult: bool
) -> tuple[list[Path] | None, list[Path]]:
    """Detect video and timestamp files in a Raw directory.

    Parameters
    ----------
    raw_folder_path : Path
        The session's Raw directory (or subfolder containing Bonsai files).
    is_adult : bool
        Whether the session is from an adult subject.

    Returns
    -------
    tuple[list[Path] | None, list[Path]]
        (video_file_paths, timestamps_file_paths). video_file_paths is None if no videos found.
    """
    if is_adult:
        video_file_paths = sorted(raw_folder_path.rglob("BonsaiVideo*.avi"))
        timestamps_file_paths = sorted(raw_folder_path.rglob("BonsaiTracking*.csv"))
    else:
        video_file_paths = sorted(
            list(raw_folder_path.rglob("BonsaiCapture*.avi"))
            + list(raw_folder_path.rglob("BonsaiVideo*.avi"))
        )
        timestamps_file_paths = sorted(
            list(raw_folder_path.rglob("Bonsai testing*.csv"))
            + list(raw_folder_path.rglob("BonsaiTracking*.csv"))
        )

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

    Handles auto-detection of stream names, file paths, and layout variations.
    Named edge-case branches handle sessions with unusual directory structures.

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
    raw_folder_path = session_folder_path / "Raw"
    processed_xml_path, nrs_path, lfp_file_path, mat_path, sleep_path = detect_processed_paths(
        session_folder_path, folder_name
    )

    if folder_name == "H3001-200202":
        # Unusual Raw subfolder: the ephys data lives in Raw/H3001-200202/
        ephys_search_root = raw_folder_path / folder_name
        raw_ephys_folder_path, raw_xml_path, stream_name, ttl_stream_name = detect_open_ephys_stream(
            ephys_search_root
        )
        video_file_paths = None
        timestamps_file_paths = sorted(ephys_search_root.rglob("Bonsai testing*.csv"))

    elif folder_name == "H3023-210812":
        # No raw Open Ephys folder; use .dat file and find media in Processed/
        processed_root = session_folder_path / "Processed"
        raw_xml_path = processed_xml_path  # processed XML used as raw XML (raw data missing)
        raw_ephys_folder_path = None
        raw_ephys_dat_file_path = processed_root / f"{folder_name}.dat"
        stream_name = None
        ttl_stream_name = None
        video_file_paths = sorted(processed_root.glob("BonsaiCapture*.avi")) or None
        timestamps_file_paths = sorted(processed_root.glob("BonsaiTracking*.csv"))
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

    elif folder_name == "H3029-230510":
        # Unusual Raw structure: Raw/day2/experiment2/; uses adult-style temporal alignment
        ephys_search_root = raw_folder_path / "day2" / "experiment2"
        raw_ephys_folder_path, raw_xml_path, stream_name, ttl_stream_name = detect_open_ephys_stream(
            ephys_search_root
        )
        video_file_paths, timestamps_file_paths = detect_video_and_timestamp_paths(
            ephys_search_root, is_adult=True
        )
        is_adult = True  # adult-style temporal alignment despite being a juvenile

    elif folder_name == "H4817-220828":
        # Raw XML is missing a channel; use processed XML as raw XML
        raw_ephys_folder_path, _, stream_name, ttl_stream_name = detect_open_ephys_stream(
            raw_folder_path
        )
        raw_xml_path = processed_xml_path
        video_file_paths, timestamps_file_paths = detect_video_and_timestamp_paths(
            raw_folder_path, is_adult
        )

    else:
        raw_ephys_folder_path, raw_xml_path, stream_name, ttl_stream_name = detect_open_ephys_stream(
            raw_folder_path
        )
        video_file_paths, timestamps_file_paths = detect_video_and_timestamp_paths(
            raw_folder_path, is_adult
        )

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
    max_workers = 1

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
