import struct
import subprocess
from pathlib import Path


def mp4_is_faststart(mp4_file_path: Path) -> bool:
    """Return True if the mp4's moov atom precedes its mdat atom (web-optimized).

    Browser players (e.g. NeuroSift on DANDI) stream over HTTP range requests and
    need the moov atom (the frame index) near the front of the file. If moov sits
    after mdat, playback fails because the player would have to download the whole
    file to reach the index. This reads only the top-level box headers, so it is
    fast even on multi-gigabyte files.
    """
    mp4_file_path = Path(mp4_file_path)
    with open(mp4_file_path, "rb") as file:
        offset = 0
        file_size = mp4_file_path.stat().st_size
        while offset < file_size:
            file.seek(offset)
            header = file.read(8)
            if len(header) < 8:
                break
            box_size, box_type = struct.unpack(">I4s", header)
            box_type = box_type.decode("ascii", errors="replace")
            if box_size == 1:
                # 64-bit largesize follows the 8-byte header.
                box_size = struct.unpack(">Q", file.read(8))[0]
            elif box_size == 0:
                # Box extends to end of file.
                box_size = file_size - offset

            if box_type == "moov":
                return True
            if box_type == "mdat":
                return False
            offset += box_size

    raise ValueError(f"No moov or mdat atom found in {mp4_file_path}")


def remux_to_faststart(mp4_file_path: Path) -> None:
    """Rewrite an mp4 in place with the moov atom moved to the front (faststart).

    This is a stream copy (``-c copy``), so no re-encoding happens: it is fast and
    lossless. The remux is written to a sibling temp file and then atomically
    renamed over the original so an interrupted run never leaves a corrupt file.
    """
    mp4_file_path = Path(mp4_file_path)
    temp_output_path = mp4_file_path.with_suffix(".faststart.mp4")
    print(f"Remuxing (faststart) {mp4_file_path.name}...")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i", str(mp4_file_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(temp_output_path),
        ],
        check=True,
    )
    temp_output_path.replace(mp4_file_path)


def convert_avi_to_mp4_h264(
    *,
    video_file_paths: list[Path],
    output_directory: Path,
) -> list[Path]:
    """Transcode each .avi (MPEG-4 part 2) to .mp4 (H.264) via system ffmpeg.

    Writes to output_directory / <source_stem>.mp4. Skips files whose output
    already exists. Returns the list of converted paths in input order.

    The ``-movflags +faststart`` flag relocates the moov atom (the frame index)
    to the front of the file so browser-based players (e.g. NeuroSift on DANDI)
    can begin playback after a single HTTP range request instead of having to
    download the entire file to find the index at the end.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    converted_paths = []
    for source_path in video_file_paths:
        source_path = Path(source_path)
        output_path = output_directory / f"{source_path.stem}.mp4"
        converted_paths.append(output_path)

        if output_path.exists():
            print(f"Skipping video conversion (already exists): {output_path}")
            continue

        print(f"Converting {source_path.name} -> {output_path.name} (H.264 MP4)...")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i", str(source_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(output_path),
            ],
            check=True,
        )

    return converted_paths
