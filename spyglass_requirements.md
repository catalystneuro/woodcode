# Spyglass Requirements for NWB Files

Spyglass is a MySQL-backed analysis pipeline built on top of NWB (NeuroData Without Borders). To be ingested via `sgi.insert_sessions(...)`, an NWB file must satisfy a number of requirements that go beyond the base NWB specification. This document lists those requirements as currently enforced by the Moore 2025 conversion pipeline.

The requirements are split into two parts:

- **Part A — Core Spyglass requirements.** Enforced by base Spyglass (`sgi.insert_sessions` and the standard Spyglass tables). Any NWB file destined for any Spyglass deployment needs these.
- **Part B — Custom extension requirements.** Imposed by the custom DataJoint tables and helper inserts added for this pipeline (`ImportedPseudoEMG`, `ImportedHistologyImages`, the sleep-stage and unit-annotation insert helpers in `moore_2025/insert_session.py`). These only apply if you use those custom tables.

---

<details>
<summary><h2 style="display:inline">Part A — Core Spyglass Requirements</h2></summary>

<details>
<summary><b>1. Raw electrophysiology</b></summary>

- **All raw electrophysiology must be written as a single `ElectricalSeries`.** Multi-segment recordings are concatenated; clock resets across segments are corrected before concatenation so timestamps remain monotonic.
- **The `ElectricalSeries` must be named exactly `e-series`** (lowercase, with a hyphen).
- The series is added to `nwbfile.acquisition`.

</details>

<details>
<summary><b>2. Epochs</b></summary>

- **Each epoch must be tagged with a two-digit zero-padded string** (`"01"`, `"02"`, `"03"`, ...). The tag drives the epoch identifier in Spyglass.

</details>

<details>
<summary><b>3. Tasks</b></summary>

- Tasks live in a processing module named `tasks`.
- Each task is its own `DynamicTable` (the table name is the task name).
- **Each task table must have exactly these columns**:
  - `task_name`
  - `task_description`
  - `task_environment`
  - `camera_id` (list of camera device IDs)
  - `task_epochs` (list of epoch indices that this task covers)

Any deviation from these column names will prevent the task from being imported.

</details>

<details>
<summary><b>4. Electrodes</b></summary>

The electrode table must include the following Spyglass-required custom columns in addition to the standard NWB ones:

- `probe_shank` — shank ID within the probe
- `probe_electrode` — electrode ID within the shank
- `bad_channel` — boolean flag for bad channels
- `ref_elect_id` — reference electrode ID

**`ref_elect_id` must always be an integer**, even when no electrode served as the original reference. Use `-1` as a sentinel in that case — `None`/missing values will fail ingestion.

</details>

<details>
<summary><b>5. Devices and extension types</b></summary>

All ephys- and camera-related devices must use the types defined by the [`ndx-franklab-novela`](https://github.com/LorenFrankLab/ndx-franklab-novela) NWB extension, in the structural arrangement that extension expects. Substituting the base NWB types (e.g. `pynwb.device.Device`, `pynwb.ecephys.ElectrodeGroup`) will not work.

The required types and their roles:

| Type | Role |
|---|---|
| `DataAcqDevice` | Data-acquisition system (amplifier, ADC, system). One per file is required. |
| `Probe` | Silicon probe. Holds a list of `Shank` objects. |
| `Shank` | One shank of a probe. Holds a list of `ShanksElectrode` objects. |
| `ShanksElectrode` | A single electrode site on a shank, with relative `(x, y, z)` coordinates. |
| `NwbElectrodeGroup` | Per-shank electrode group. Holds targeted stereotaxic coordinates and the parent `Probe`. |
| `CameraDevice` | Behavioral camera. Referenced by `camera_id` in the task table and by `ImageSeries.device` for video. **Must be named `"camera_device N"`** where `N` is an integer (e.g. `"camera_device 1"`, `"camera_device 2"`). |

Hierarchy: `Probe` contains `Shank`s, each `Shank` contains `ShanksElectrode`s. Each shank also has a corresponding `NwbElectrodeGroup` that the electrode-table rows belong to.

</details>

<details>
<summary><b>6. LFP</b></summary>

- LFP must be written as an `ElectricalSeries` named exactly **`LFP`**, wrapped in a `pynwb.ecephys.LFP` container.
- The `LFP` container must live in the `ecephys` processing module (i.e. `nwbfile.processing["ecephys"]`).

</details>

<details>
<summary><b>7. Position and head direction (behavior module)</b></summary>

- Position and head-direction data live in the `behavior` processing module (i.e. `nwbfile.processing["behavior"]`).
- Position is a `pynwb.behavior.Position` object containing one or more `SpatialSeries` objects. Each `SpatialSeries` becomes a row in Spyglass's `PositionSource.SpatialSeries` table — the SpatialSeries names are not fixed, but every name must be unique within the Position object.
- Head direction is a `pynwb.behavior.CompassDirection` object containing a `SpatialSeries`. Drives Spyglass's `RawCompassDirection` table.

</details>

<details>
<summary><b>8. Video</b></summary>

- Each video file is a `pynwb.image.ImageSeries` added to `nwbfile.acquisition`.
- The `ImageSeries.device` field must point to a `CameraDevice` (see §5). The link to `CameraDevice` is what drives Spyglass's `VideoFile` table.

</details>

</details>

---

<details>
<summary><h2 style="display:inline">Part B — Custom Extension Requirements</h2></summary>

These requirements are imposed by the custom Spyglass tables in [moore_2025/spyglass_extensions/](moore_2025/spyglass_extensions/) and the helper inserts in [moore_2025/insert_session.py](moore_2025/insert_session.py). They are *not* enforced by base Spyglass — they only apply when using these custom tables in this pipeline.

<details>
<summary><b>9. Pseudo-EMG (<code>ImportedPseudoEMG</code>)</b></summary>

- A `pynwb.TimeSeries` named exactly **`pseudoEMG`** must live in the `ecephys` processing module (i.e. `nwbfile.processing["ecephys"]["pseudoEMG"]`). Read by `ImportedPseudoEMG.make()`.

</details>

<details>
<summary><b>10. Histology images (<code>ImportedHistologyImages</code>)</b></summary>

- A `pynwb.image.Images` container named exactly **`histology_images`** must be added to `nwbfile.acquisition`.
- Each image inside is exposed as its own row in `ImportedHistologyImages` keyed by the image's name, so individual image names must be unique within the container.

</details>

<details>
<summary><b>11. Sleep stages (custom <code>insert_sleep</code> helper)</b></summary>

- Sleep-stage intervals must be added as a `TimeIntervals` table named exactly **`sleep_stages`** via `nwbfile.add_time_intervals(...)` (so it lands at `nwbfile.intervals["sleep_stages"]`).
- Each row's `tags` column must be one of the literal values **`["rem"]`**, **`["nrem"]`**, or **`["wake"]`**. Other tag values are silently ignored by `insert_sleep()`.

</details>

<details>
<summary><b>12. Units table (custom unit annotations via <code>insert_sorting</code>)</b></summary>

The standard NWB units table must additionally carry:

- A **`sampling_rate`** column — sampling rate of the raw ephys signal that produced the unit.
- A **`waveform_mean`** value per unit — the mean waveform array.

Both are pushed into Spyglass's `ImportedSpikeSorting.Annotations` table by the `insert_sorting()` helper.

</details>

</details>
