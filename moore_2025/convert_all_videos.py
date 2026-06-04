"""Standalone pass that transcodes every session's Bonsai .avi video to H.264 MP4.

The main conversion (``convert_all_sessions.py``) transcodes video inline via
``nwb.video_codec.convert_avi_to_mp4_h264``, which is slow and competes with the
parallel per-session workers. Because that function skips any ``.mp4`` that
already exists in the output directory, running this script first writes all the
transcoded videos into ``output_dir_path`` so the main conversion's video step
becomes a fast skip (it still references the already-converted ``.mp4`` files
when building each ImageSeries).

Run this as a separate pass before (or independently of) ``convert_all_sessions.py``.
Transcoding here is fully serial (one ``ffmpeg`` subprocess at a time) so it is
robust and decoupled from the session ProcessPool.
"""
from pathlib import Path

from woodcode.nwb.video_codec import convert_avi_to_mp4_h264
from moore_2025.convert_all_sessions import (
    collect_juvenile_session_to_nwb_kwargs,
    collect_adult_session_to_nwb_kwargs,
)


def collect_all_video_file_paths(
    *,
    juvenile_dir_path: Path,
    adult_dir_path: Path,
    meta_path: Path,
    juvenile_histology_folder_path: Path,
    adult_histology_folder_path: Path,
    juvenile_metadata_file_path: Path,
    adult_metadata_file_path: Path,
) -> list[Path]:
    """Collect every session's Bonsai .avi video paths across both cohorts.

    Reuses the same kwargs-collection logic as the main conversion so that
    exactly the videos the main conversion would transcode are gathered here.
    Sessions without video contribute nothing.

    Returns
    -------
    list[Path]
        A flat list of .avi video file paths across all sessions.
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

    video_file_paths = []
    for session_to_nwb_kwargs in session_to_nwb_kwargs_per_session:
        session_video_file_paths = session_to_nwb_kwargs["video_file_paths"]
        if session_video_file_paths:
            video_file_paths.extend(session_video_file_paths)
    return video_file_paths


def convert_all_videos(
    *,
    juvenile_dir_path: Path,
    adult_dir_path: Path,
    output_dir_path: Path,
    meta_path: Path,
    juvenile_histology_folder_path: Path,
    adult_histology_folder_path: Path,
    juvenile_metadata_file_path: Path,
    adult_metadata_file_path: Path,
):
    """Transcode every session's video to H.264 MP4 in a standalone pass.

    Each .avi is transcoded to ``output_dir_path / <stem>.mp4``, one at a time.
    Outputs that already exist are skipped, so this pass is resumable: re-running
    only transcodes the videos that are still missing.

    Parameters
    ----------
    juvenile_dir_path : Path
        Root directory containing WT/ and KO/ subdirectories for juvenile sessions.
    adult_dir_path : Path
        Root directory containing WT/ and KO/ subdirectories for adult sessions.
    output_dir_path : Path
        Directory where the transcoded .mp4 files are written. Must match the
        ``output_dir_path``/``save_path`` used by the main conversion so that the
        main conversion skips re-transcoding.
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
    """
    output_dir_path.mkdir(parents=True, exist_ok=True)
    video_file_paths = collect_all_video_file_paths(
        juvenile_dir_path=juvenile_dir_path,
        adult_dir_path=adult_dir_path,
        meta_path=meta_path,
        juvenile_histology_folder_path=juvenile_histology_folder_path,
        adult_histology_folder_path=adult_histology_folder_path,
        juvenile_metadata_file_path=juvenile_metadata_file_path,
        adult_metadata_file_path=adult_metadata_file_path,
    )
    print(f"Found {len(video_file_paths)} video files to transcode.")

    convert_avi_to_mp4_h264(video_file_paths=video_file_paths, output_directory=output_dir_path)


if __name__ == "__main__":
    juvenile_dir_path = Path("/Volumes/SamsungSSD/CatalystNeuro/Dudchenko/251104_MooreDataset/H3000_Juveniles")
    adult_dir_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/H4800_Adults")
    meta_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/MooreDataset_Metadata.xlsx")
    juvenile_histology_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/Histology/H3000")
    adult_histology_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/Histology/H4800")
    output_dir_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw")
    juvenile_metadata_file_path = Path(__file__).parent / "juvenile_metadata.yaml"
    adult_metadata_file_path = Path(__file__).parent / "adult_metadata.yaml"

    convert_all_videos(
        juvenile_dir_path=juvenile_dir_path,
        adult_dir_path=adult_dir_path,
        output_dir_path=output_dir_path,
        meta_path=meta_path,
        juvenile_histology_folder_path=juvenile_histology_folder_path,
        adult_histology_folder_path=adult_histology_folder_path,
        juvenile_metadata_file_path=juvenile_metadata_file_path,
        adult_metadata_file_path=adult_metadata_file_path,
    )
