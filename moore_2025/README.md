# Moore 2025 NWB Conversion

Conversion scripts for the Moore 2025 dataset (Wood/Dudchenko lab) to the
[Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format, and ingestion of the resulting
NWB files into a [Spyglass](https://lorenfranklab.github.io/spyglass/) analysis database.

The dataset consists of freely-moving electrophysiology, position tracking, behavioral video, sleep scoring, and
histology from juvenile and adult mice performing a foraging task. The conversion lives in the
[moore_2025/](.) directory of the [`woodcode`](https://github.com/catalystneuro/woodcode) package and is built on
the reusable NWB-building helpers in [woodcode/nwb/](../woodcode/nwb/).

## Two-Stage Pipeline

This project has two distinct stages, which run in **two separate conda environments** because Spyglass and
NeuroConv have conflicting dependencies:

1. **Conversion** (environment `dudchenko_lab_to_nwb_env`): reads the raw acquisition files and writes one `.nwb` file per
   session. This stage uses NeuroConv/SpikeInterface and the `woodcode` package. Scripts:
   [convert_single_session.py](convert_single_session.py), [convert_example_sessions.py](convert_example_sessions.py),
   [convert_edge_case_sessions.py](convert_edge_case_sessions.py), and
   [convert_all_sessions.py](convert_all_sessions.py).

2. **Spyglass ingestion** (a separate environment built from the Spyglass install instructions): reads the `.nwb`
   files produced in stage 1 and inserts them into a MySQL-backed Spyglass database. Scripts:
   [insert_single_session.py](insert_single_session.py), [insert_example_sessions.py](insert_example_sessions.py),
   [insert_edge_case_sessions.py](insert_edge_case_sessions.py), and
   [insert_all_sessions.py](insert_all_sessions.py).

You must finish stage 1 before starting stage 2.

## Installation

### Conversion environment (`dudchenko_lab_to_nwb_env`)

We recommend installing directly from GitHub so you can amend the source code to adapt to future experimental
differences. You will need `git` ([installation instructions](https://github.com/git-guides/install-git)) and
`conda` ([installation instructions](https://docs.conda.io/en/latest/miniconda.html)).

```bash
git clone https://github.com/catalystneuro/woodcode
cd woodcode
conda env create --file make_env.yml
conda activate dudchenko_lab_to_nwb_env
pip install ndx-franklab-novela --no-deps
```

This creates a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)
that isolates the conversion code from your system libraries. The environment installs the `woodcode` package in
[editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs), so any changes you make to the
source take effect immediately.

The extra `pip install ndx-franklab-novela --no-deps` is required because `ndx-franklab-novela` pins an older
`hdmf` that conflicts with the development build of `pynwb` this conversion depends on. Installing it with
`--no-deps` adds the extension without letting it drag in that incompatible `hdmf`, which is why it is kept out of
`make_env.yml`/`pyproject.toml` and installed as a separate step.

The [make_env.yml](../make_env.yml) file additionally installs the **`ffmpeg`** system binary, which is not a
Python package and so cannot be declared in `pyproject.toml`. It is used by
[woodcode/nwb/video_codec.py](../woodcode/nwb/video_codec.py) to transcode Bonsai `.avi` video to H.264 `.mp4`.

### Spyglass environment

The Spyglass ingestion scripts require a **separate** environment, because Spyglass is not compatible with the
NeuroConv stack used for conversion. Build it by following the official
[Spyglass installation instructions](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/), which also
cover standing up the required MySQL database. The insertion scripts do not require the `woodcode` package to be
installed in this environment — they reach the custom tables in [spyglass_extensions/](spyglass_extensions/) via a
`sys.path.append(...)` near the top of [session_to_spyglass.py](session_to_spyglass.py).

The insertion scripts expect a DataJoint config file at the path hard-coded near the top of
[session_to_spyglass.py](session_to_spyglass.py) (`dj_local_conf.json`). Update that path for your machine.

## Helpful Definitions

Unlike a typical NeuroConv conversion, this project does **not** use `DataInterface` / `NWBConverter` classes.
Instead, the reusable NWB-building logic is a library of `add_*` functions in
[woodcode/nwb/convert.py](../woodcode/nwb/convert.py), and the dataset-specific
[session_to_nwb.py](session_to_nwb.py) orchestrates them. The conceptual roles are still the same:

- **Readers** ([woodcode/nwb/io.py](../woodcode/nwb/io.py)) parse a single source-data modality (Neuroscope `.xml`,
  the metadata `.xlsx`, MATLAB `.mat` tracking/spikes, binary `.dat`/`.lfp`, etc.) into numpy/pynapple objects.
- **Adders** ([woodcode/nwb/convert.py](../woodcode/nwb/convert.py)) write each modality into the NWB file
  (`add_raw_ephys`, `add_lfp`, `add_units`, `add_tracking`, `add_video`, `add_sleep`, `add_histology`, ...).
- **Orchestration** ([session_to_nwb.py](session_to_nwb.py)) defines `session_to_nwb()`, which calls the readers
  and adders in the correct order and handles temporal alignment for a single session.
- **Batch driver** ([convert_all_sessions.py](convert_all_sessions.py)) defines `dataset_to_nwb()`, which discovers
  every session, builds the per-session arguments, and converts them in parallel.

## Repository Structure

```
woodcode/
├── make_env.yml                          # conda environment for the conversion stage
├── pyproject.toml
├── spyglass_requirements.md              # NWB-structure constraints that Spyglass enforces
├── tables_*.txt                          # Spyglass table dumps, one per example session (QA artifacts)
├── woodcode/
│   └── nwb/                              # reusable, lab-generic NWB-building helpers
│       ├── convert.py                   # add_* functions that write each modality into NWB
│       ├── io.py                        # readers/parsers for source files (xml, xlsx, mat, dat/lfp)
│       ├── edit.py                      # small post-hoc NWB edits (e.g. change_genotype)
│       ├── video_codec.py               # ffmpeg-based .avi -> .mp4 (H.264) transcoding
│       ├── dat_file_data_chunk_iterator.py
│       └── multi_segment_recording_data_chunk_iterator.py
└── moore_2025/                          # the Moore 2025 conversion (this directory)
    ├── session_to_nwb.py                # session_to_nwb(): convert a single session + get_probe_info_*
    ├── convert_single_session.py        # convert the first example session (fast smoke test)
    ├── convert_example_sessions.py      # convert the six representative example sessions
    ├── convert_edge_case_sessions.py    # convert the edge-case sessions
    ├── convert_all_sessions.py          # dataset_to_nwb(): convert the whole dataset in parallel
    ├── convert_all_videos.py            # (optional) transcode every session's video in a separate serial pass
    ├── temporal_alignment.py            # align video timestamps to the ephys clock
    ├── adult_metadata.yaml              # cohort-constant metadata for adult mice
    ├── juvenile_metadata.yaml           # cohort-constant metadata for juvenile mice
    ├── session_to_spyglass.py           # insert_session(): ingest one NWB file into Spyglass + helpers
    ├── insert_single_session.py         # ingest the first example session (fast smoke test)
    ├── insert_example_sessions.py       # ingest the six representative example sessions
    ├── insert_edge_case_sessions.py     # ingest the edge-case sessions
    ├── insert_all_sessions.py           # ingest all NWB files into Spyglass
    ├── example_notebook.ipynb           # example analysis of a converted/ingested session
    └── spyglass_extensions/             # custom DataJoint tables
        ├── imported_pseudo_emg.py       # ImportedPseudoEMG table
        └── imported_histology_images.py # ImportedHistologyImages table
```

Inside the [moore_2025/](.) directory you will find the following files:

* [session_to_nwb.py](session_to_nwb.py) : Defines `session_to_nwb()`, which converts a single session of data to
  NWB. It also defines the `get_probe_info_*` helpers that build probe geometry from the
  [probeinterface library](https://github.com/SpikeInterface/probeinterface_library) for the Cambridge Neurotech
  H5, H6b, and H7 probes. This is the reusable module imported by the driver scripts below and by
  [convert_all_sessions.py](convert_all_sessions.py).

* [convert_single_session.py](convert_single_session.py) / [convert_example_sessions.py](convert_example_sessions.py)
  / [convert_edge_case_sessions.py](convert_edge_case_sessions.py) : Driver scripts that call `session_to_nwb()` on
  hand-curated sessions chosen to exercise every code path (juvenile/adult, WT/KO, and the edge cases described
  below). `convert_single_session.py` converts only the first example session and is the fastest way to smoke-test
  the pipeline; `convert_example_sessions.py` converts the six representative sessions (juvenile WT/KO across two
  days, plus adult WT/KO); `convert_edge_case_sessions.py` converts the remaining edge cases. Each script defines
  its own data-path constants near the top of `main()`.

* [convert_all_sessions.py](convert_all_sessions.py) : Defines `dataset_to_nwb()`, which converts every session in
  the dataset. It auto-discovers sessions from the `WT`/`KO` subfolders, builds the arguments for each, and runs them
  through a `ProcessPoolExecutor`. When run as a script, its `main()` calls `dataset_to_nwb()` with the appropriate
  arguments. This module also holds the per-session lookup tables (`STREAM_NAME_PER_SESSION`,
  `SESSIONS_WITHOUT_RAW_DATA`, `SESSIONS_WITHOUT_VIDEO`, `SESSIONS_WITHOUT_RAW_BONSAI_OUTPUT`,
  `SESSIONS_USING_PROCESSED_XML`, `SESSION_TO_ALT_XML_SESSION`, `SESSIONS_TO_SKIP`) that encode every known
  deviation from the default data layout — this is the main place to edit when adding new sessions.

* [convert_all_videos.py](convert_all_videos.py) : **Optional** helper that runs only the video-transcoding step of
  the conversion. It reuses the same session-discovery logic as `convert_all_sessions.py` to find every session's
  Bonsai `.avi`, then transcodes each to H.264 `.mp4` in the output directory, one at a time. This is useful because
  transcoding is the slowest part of the conversion: running it as a separate serial pass keeps the heavy `ffmpeg`
  work out of the parallel `convert_all_sessions.py` workers. Because `video_codec.convert_avi_to_mp4_h264` skips any
  `.mp4` that already exists, a subsequent `convert_all_sessions.py` run finds the videos already transcoded and just
  references them. The pass is resumable for the same reason. See [Convert all sessions](#convert-all-sessions).

* [temporal_alignment.py](temporal_alignment.py) : Puts the video camera clock onto the electrophysiology clock.
  Adult sessions use a Basler camera that emits one TTL pulse per frame, so alignment is a near 1:1 TTL-to-frame
  mapping (correcting for dropped frames and clock resets). Juvenile sessions use an Arduino that sends a random
  pulse sequence to both the ephys board (as TTL) and an LED in the camera's field of view; the random interval
  pattern is used as a fingerprint to match LED events (video clock) to TTL events (ephys clock), and intermediate
  frames are interpolated. The conversion calls `get_aligned_video_timestamps_adults()`,
  `get_aligned_video_timestamps_juveniles()`, `get_aligned_video_timestamps_juveniles_from_dat()`, and
  `get_start_time()`.

* [adult_metadata.yaml](adult_metadata.yaml) / [juvenile_metadata.yaml](juvenile_metadata.yaml) : Cohort-constant
  metadata that is deep-merged into the per-session metadata read from the `MooreDataset_Metadata.xlsx` spreadsheet.
  They contain the task descriptions and environment, the camera device (a ceiling-mounted Basler for adults vs. a
  Logitech for juveniles), and the probe/data-acquisition system. Per-subject, per-session values (e.g. genotype)
  live in the spreadsheet, not here.

* [session_to_spyglass.py](session_to_spyglass.py) : Defines `insert_session()`, which ingests a single converted
  NWB file into Spyglass via `sgi.insert_sessions(...)` and then runs the custom inserts (`insert_sleep`,
  `insert_sorting`, `insert_pseudo_emg`, `insert_histology_images`). It also defines `print_tables()` (dumps the
  resulting tables to a `tables_*.txt` file for verification) and `clear_shared_tables()` (clears the shared
  probe/camera/device/task records before a fresh insertion). This is the reusable module imported by the driver
  scripts below and by [insert_all_sessions.py](insert_all_sessions.py).

* [insert_single_session.py](insert_single_session.py) / [insert_example_sessions.py](insert_example_sessions.py)
  / [insert_edge_case_sessions.py](insert_edge_case_sessions.py) : Driver scripts that ingest the same curated
  sessions as the stage-1 convert drivers. Each calls `clear_shared_tables()` once, then ingests its sessions and
  dumps a `tables_*.txt` per session. `insert_single_session.py` ingests only the first example session (fastest
  smoke test); `insert_example_sessions.py` ingests the six representative sessions; `insert_edge_case_sessions.py`
  ingests the edge cases. Each defines its own `raw_data_path` near the top of `main()`.

* [insert_all_sessions.py](insert_all_sessions.py) : Batch version that ingests every `.nwb` file in the Spyglass
  raw directory.

* [spyglass_extensions/](spyglass_extensions/) : Custom DataJoint `dj.Imported` tables for data that base Spyglass
  does not model — `ImportedPseudoEMG` (a muscle-activity proxy derived from the ephys signal) and
  `ImportedHistologyImages` (histology images stored in the NWB file).

See [spyglass_requirements.md](../spyglass_requirements.md) for the full list of NWB-structure constraints that
Spyglass enforces (e.g. the raw `ElectricalSeries` must be named exactly `e-series`, epochs must be tagged with
two-digit strings, etc.). These constraints shape how [woodcode/nwb/convert.py](../woodcode/nwb/convert.py) writes
the file.

## Running the Conversion (Stage 1)

Activate the conversion environment first:

```bash
conda activate dudchenko_lab_to_nwb_env
```

### Convert example sessions

The example conversion is split across three driver scripts, all of which call `session_to_nwb()` from
[session_to_nwb.py](session_to_nwb.py):
- [convert_single_session.py](convert_single_session.py) — converts only the first example session. This is the
  fastest way to smoke-test the pipeline end-to-end.
- [convert_example_sessions.py](convert_example_sessions.py) — converts the six representative sessions.
- [convert_edge_case_sessions.py](convert_edge_case_sessions.py) — converts the edge-case sessions.

1. In each script you intend to run, update the path variables at the top of `main()` to point to your local
   data. These are currently hard-coded to the original author's external drives and must be changed (the same
   block appears in all three scripts):
   - `juvenile_folder_path` — the `H3000_Juveniles` data root
   - `adult_folder_path` — the `H4800_Adults` data root
   - `meta_path` — the `MooreDataset_Metadata.xlsx` spreadsheet
   - `histology_folder_path` — the `Histology` folder (with `H3000`/`H4800` subfolders)
   - `output_folder_path` — where the `.nwb` files are written
   - `juvenile_metadata_file_path` / `adult_metadata_file_path` — the two YAML files in this directory

2. Run the script you need, e.g. the fast single-session smoke test:
    ```bash
    python -W ignore moore_2025/convert_single_session.py
    ```

   The scripts do not clear `output_folder_path`; converted files accumulate there and per-session outputs are
   overwritten, so you can run the scripts in sequence into one folder. Prefer `convert_single_session.py` for a
   quick check — `stub_test` is not an effective speed-up here because temporal alignment still runs on the full
   data.

### Convert all sessions

1. Update the same data paths in the `__main__` block of [convert_all_sessions.py](convert_all_sessions.py)
   (`juvenile_dir_path`, `adult_dir_path`, `meta_path`, `juvenile_histology_folder_path`,
   `adult_histology_folder_path`, `output_dir_path`). The two metadata YAML paths resolve automatically relative to
   the script. You can also set `stub_test` and `max_workers` (the number of parallel processes).

   Like the example driver scripts, this script does **not** delete the output directory first.

2. Run the conversion script:
    ```bash
    python -W ignore moore_2025/convert_all_sessions.py
    ```

   Conversion is fault-tolerant: if a session fails, the error and its traceback are written to
   `ERROR_<session>.txt` in the output directory and the batch continues. After a run, check the output directory
   for `ERROR_*.txt` files to find sessions that did not convert.

#### (Optional) Pre-transcode video in a separate pass

Transcoding the Bonsai `.avi` video to H.264 `.mp4` is the slowest part of the conversion, and running it inside the
parallel `convert_all_sessions.py` workers is the heaviest step. If you prefer, you can do all the transcoding up
front in a single serial pass with [convert_all_videos.py](convert_all_videos.py):

1. Update the data paths in its `__main__` block (the same `juvenile_dir_path`, `adult_dir_path`, `meta_path`,
   `juvenile_histology_folder_path`, `adult_histology_folder_path`, and `output_dir_path` as
   `convert_all_sessions.py`). **`output_dir_path` must match** the `convert_all_sessions.py` output directory so the
   main conversion finds the transcoded files.

2. Run it:
    ```bash
    python -W ignore moore_2025/convert_all_videos.py
    ```

It writes one `.mp4` per video into `output_dir_path`. A subsequent `convert_all_sessions.py` run then skips
transcoding (it reuses any `.mp4` that already exists) and just references the converted files when building each
`ImageSeries`. The pass is resumable: re-running only transcodes the videos that are still missing. This step is
purely optional — `convert_all_sessions.py` transcodes any missing video itself.

## Running the Spyglass Ingestion (Stage 2)

Switch to the Spyglass environment first (see [Installation](#installation)). Make sure stage 1 has produced the
`.nwb` files and that the MySQL database and `dj_local_conf.json` are configured.

### Insert example sessions

The example ingestion is split across three driver scripts, all of which import `insert_session()` and the
helpers from [session_to_spyglass.py](session_to_spyglass.py):
- [insert_single_session.py](insert_single_session.py) — ingests only the first example session (fastest check).
- [insert_example_sessions.py](insert_example_sessions.py) — ingests the six representative sessions.
- [insert_edge_case_sessions.py](insert_edge_case_sessions.py) — ingests the edge-case sessions.

1. In [session_to_spyglass.py](session_to_spyglass.py), update `dj_local_conf_path` and the `sys.path.append(...)`
   to `spyglass_extensions`. In each driver script you intend to run, update `raw_data_path` (the directory
   containing the `.nwb` files).

2. Run the script you need, e.g. the fast single-session smoke test:
    ```bash
    python moore_2025/insert_single_session.py
    ```

   Each driver first calls `clear_shared_tables()`, which deletes the shared probe/camera/device/task entries
   it is about to re-insert, then inserts its sessions and writes a `tables_*.txt` dump per session for
   verification. Because deleting the probe types cascades, each script resets the shared tables on every run, so
   running one driver after another re-ingests from a clean slate rather than accumulating.

### Insert all sessions

1. In [insert_all_sessions.py](insert_all_sessions.py), update `dj_local_conf_path`, the `sys.path.append(...)`, and
   `spyglass_raw_path` (the directory of `.nwb` files to ingest).

   > [!WARNING]
   > `insert_all_sessions.py` calls `sgc.Nwbfile.delete()` to wipe the database for a clean full rebuild before
   > re-inserting every session. Only run it when you intend to rebuild the whole database.

2. Run the script:
    ```bash
    python moore_2025/insert_all_sessions.py
    ```

## Adapting the Conversion for New Sessions

When new sessions are added to the dataset, most adjustments happen in
[convert_all_sessions.py](convert_all_sessions.py) and the metadata files — not in the core
[woodcode/nwb/](../woodcode/nwb/) helpers.

- **Per-session metadata** (subject, genotype, probe model, channel maps, etc.) comes from the
  `MooreDataset_Metadata.xlsx` spreadsheet. Add a row there for each new session. The probe model string in that
  spreadsheet (read as `metadata["probe"][0]["type"]`) selects the probe geometry.

- **Cohort-constant metadata** (task descriptions, camera hardware, `meters_per_pixel`, probe reference text) lives
  in [adult_metadata.yaml](adult_metadata.yaml) / [juvenile_metadata.yaml](juvenile_metadata.yaml). Edit these for
  changes that apply to a whole cohort, or add new `Task` / `Video` entries if new epoch or video types appear.

- **Layout deviations** are encoded in the lookup tables in [convert_all_sessions.py](convert_all_sessions.py). For
  a new session that differs from the default directory/stream layout, add it to the appropriate table:
  - `STREAM_NAME_PER_SESSION` — the Open Ephys stream name (the TTL stream is always `<stream_name>_ADC`). Set to
    `None` for sessions that only have a processed `.dat` file.
  - `SESSIONS_WITHOUT_RAW_DATA` — only the processed `.dat`/`.lfp` are available.
  - `SESSIONS_WITHOUT_VIDEO` — no video was recorded.
  - `SESSIONS_WITHOUT_RAW_BONSAI_OUTPUT` — the raw Bonsai tracking CSV cannot be aligned (TTLs don't match, files
    missing, or no LED flashes).
  - `SESSIONS_USING_PROCESSED_XML` — use `Processed/<session>.xml` even when raw data exists (e.g. the raw XML is
    missing a channel).
  - `SESSION_TO_ALT_XML_SESSION` — borrow another session's XML when this session's XMLs are unusable.
  - `SESSIONS_TO_SKIP` — skip a session entirely.

- **New probe models** require a new `get_probe_info_*` helper in [session_to_nwb.py](session_to_nwb.py) (look up
  the probe in the [probeinterface library](https://github.com/SpikeInterface/probeinterface_library)) plus a branch
  in `get_probe_info()`.

- **New acquisition hardware** (a different camera, LED brightness, TTL voltage, frame rate, or Bonsai CSV column
  layout) requires updating the calibrated constants in [temporal_alignment.py](temporal_alignment.py) (LED/TTL
  thresholds, ADC channel names, `video_sampling_rate`, cooldown, `min_matches`, tolerance, and the `Item*` column
  names).

- **New probe/camera/device names** must also be added to the cleanup `delete()` blocks in
  `clear_shared_tables()` in [session_to_spyglass.py](session_to_spyglass.py) and in
  [insert_all_sessions.py](insert_all_sessions.py).

## Understanding the Data

The dataset contains freely-moving recordings from juvenile (`H3000`) and adult (`H4800`) mice, each split into
wild-type (`WT`) and knockout (`KO`) groups. Each session includes some or all of:

- **Raw electrophysiology** from Cambridge Neurotech silicon probes (H5, H6b, or H7), recorded with Open Ephys
  (Intan headstage). Stored as a single `e-series` `ElectricalSeries`.
- **LFP** downsampled from the raw signal.
- **Spike-sorted units** with mean waveforms, organized per shank.
- **Position tracking and head direction** processed in MATLAB, plus the raw Bonsai LED tracking.
- **Behavioral video** (transcoded to H.264 `.mp4`) synchronized to the ephys clock.
- **Sleep scoring** (REM / NREM / WAKE intervals) and a pseudo-EMG movement proxy.
- **Recording epochs** corresponding to the `wake`, `sleep`, `wake_cue_rot`, and `error` tasks.
- **Histology images**.

Data is organized by cohort and genotype (`H3000_Juveniles/WT`, `H4800_Adults/KO`, ...) with one folder per session
named `<subject>-<date>` (e.g. `H4813-220728`). Each session folder contains a `Raw/` tree (Open Ephys output) and a
`Processed/` tree (Neuroscope `.xml`/`.nrs`/`.lfp`/`.dat`, MATLAB tracking/spikes, sleep scoring, and Bonsai video).

### Edge cases in the dataset

The dataset includes several sessions that deviate from the standard layout; these are the sessions in
`convert_edge_case_sessions.py:main()` and are documented by the lookup tables described above:

- Sessions with **no video** (raw Bonsai tracking cannot be aligned).
- Sessions with **no raw Open Ephys output**, converted from the processed `.dat` file instead.
- Sessions with **no video and no raw output** at all.
- A **juvenile session that uses adult-style alignment** (and an H5 probe).
- Sessions with an **error epoch** (a failed recording segment).
- A session with a **clock reset** between recording segments.

## Tips for Using the Code

- **Start with the example sessions.** The `main()` functions in the
  [convert_single_session.py](convert_single_session.py) / [convert_example_sessions.py](convert_example_sessions.py)
  / [convert_edge_case_sessions.py](convert_edge_case_sessions.py) driver scripts and in their stage-2
  counterparts ([insert_single_session.py](insert_single_session.py) /
  [insert_example_sessions.py](insert_example_sessions.py) /
  [insert_edge_case_sessions.py](insert_edge_case_sessions.py)) demonstrate the expected data organization,
  parameter values, and every edge case in the dataset.

- **Smoke-test with a single session.** Run [convert_single_session.py](convert_single_session.py) (stage 1) or
  [insert_single_session.py](insert_single_session.py) (stage 2) for the fastest end-to-end check. `stub_test` is
  not an effective speed-up because temporal alignment still runs on the full data.

- **Check your file paths.** The conversion fails loudly if input files are missing. For batch runs, inspect the
  `ERROR_*.txt` files written to the output directory.

- **Mind the destructive steps.** `insert_all_sessions.py` wipes the Spyglass database. Read the warnings above
  before running it. (The conversion driver scripts do not delete their output directory.)

- **Edit metadata, not code, for descriptive changes.** The `.yaml` files and the metadata spreadsheet hold
  experiment metadata; you can update them without touching the Python.

- **Read the docstrings.** Functions and classes have NumPy-style docstrings. Use `help(function_name)` in Python to
  view them.

## Getting Help

1. Check the docstrings in the code for detailed information about each function.
2. Review the example sessions in the convert driver scripts
   ([convert_single_session.py](convert_single_session.py) /
   [convert_example_sessions.py](convert_example_sessions.py) /
   [convert_edge_case_sessions.py](convert_edge_case_sessions.py)) and their insert counterparts
   ([insert_single_session.py](insert_single_session.py) / [insert_example_sessions.py](insert_example_sessions.py)
   / [insert_edge_case_sessions.py](insert_edge_case_sessions.py)).
3. See [spyglass_requirements.md](../spyglass_requirements.md) for the NWB-structure constraints Spyglass enforces.
4. Consult the [NWB documentation](https://nwb-overview.readthedocs.io/) for the NWB format,
   the [NeuroConv documentation](https://neuroconv.readthedocs.io/) for data interfaces, and the
   [Spyglass documentation](https://lorenfranklab.github.io/spyglass/) for the analysis database.
