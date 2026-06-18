"""Quick structural/timebase verification of the converted Duszkiewicz NWB (stub or full)."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO

nwb_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw/H6813-240605.nwb")

with NWBHDF5IO(str(nwb_path), "r") as io:
    nwbfile = io.read()

    print("=== SUBJECT / SESSION ===")
    print("identifier:", nwbfile.identifier, "| session_id:", nwbfile.session_id)
    print("subject:", nwbfile.subject.subject_id, nwbfile.subject.species, nwbfile.subject.genotype, nwbfile.subject.sex)
    print("session_start_time:", nwbfile.session_start_time)

    print("\n=== PROBES / ELECTRODE GROUPS ===")
    for name, group in nwbfile.electrode_groups.items():
        print(f"  {name}: location={group.location} device={group.device.name}")
    print("n_electrodes:", len(nwbfile.electrodes))

    print("\n=== RAW E-SERIES TIMEBASE ===")
    es = nwbfile.acquisition["e-series"]
    ts = es.timestamps[:] if es.timestamps is not None else None
    if ts is not None:
        d = np.diff(ts)
        gap_i = int(np.argmax(d))
        print(f"  n={len(ts)} first={ts[0]:.4f} last={ts[-1]:.4f}  largest gap {d[gap_i]:.2f}s at [{ts[gap_i]:.3f} -> {ts[gap_i+1]:.3f}]")
    else:
        print(f"  rate={es.rate} starting_time={es.starting_time}")

    print("\n=== EPOCHS ===")
    print(nwbfile.epochs.to_dataframe()[["start_time", "stop_time", "tags"]])

    print("\n=== TASKS ===")
    if "tasks" in nwbfile.processing:
        for n in nwbfile.processing["tasks"].data_interfaces:
            print(" ", n)

    print("\n=== CUE EPOCHS (TimeIntervals) ===")
    for n in ["epCue1", "epCue2", "epCue3", "epCue4"]:
        if n in nwbfile.intervals:
            df = nwbfile.intervals[n].to_dataframe()
            print(f"  {n}: {len(df)} rows, start range [{df.start_time.min():.2f}, {df.start_time.max():.2f}]")

    print("\n=== behavioral_events (Spyglass DIO) ===")
    be = nwbfile.processing["behavior"].data_interfaces.get("behavioral_events")
    if be is not None:
        for tsname, series in be.time_series.items():
            t = series.timestamps[:]
            print(f"  {tsname}: n={len(t)} t=[{t.min():.3f}, {t.max():.3f}] ascending={np.all(np.diff(t) >= 0)}")
    else:
        print("  MISSING behavioral_events container!")

    print("\n=== OTHER ===")
    print("acquisition:", list(nwbfile.acquisition.keys()))
    print("processing[behavior]:", list(nwbfile.processing["behavior"].data_interfaces.keys()))
    print("processing[ecephys]:", list(nwbfile.processing["ecephys"].data_interfaces.keys()))
    print("n_units:", len(nwbfile.units) if nwbfile.units is not None else 0)
    print("intervals:", list(nwbfile.intervals.keys()))
