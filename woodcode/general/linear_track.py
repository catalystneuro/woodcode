import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt


def get_lap_intervals(track_pos_tsd, track_end=0.15):
    """
    Identify run times and directions of a rat moving on a track.

    Parameters:
    track_pos_tsd (nap.Tsd): Pynapple Tsd object containing position data.
    track_end (float, optional): Threshold for considering the rat at the left and right end. Default is 0.15.

    Returns:
    run_intervals (dict): Dictionary containing 'rightward' and 'leftward' nap.IntervalSet objects.
    """
    # Convert to numpy arrays immediately for faster operations
    track_pos_df = track_pos_tsd.as_series().dropna()
    track_pos = track_pos_df.values
    timestamps = track_pos_df.index.values.astype(float)

    if len(track_pos) < 2:  # Need at least 2 points for a lap
        return {"rightward": nap.IntervalSet(), "leftward": nap.IntervalSet()}

    # Vectorized normalization
    min_pos, max_pos = track_pos.min(), track_pos.max()
    if min_pos == max_pos:
        return {"rightward": nap.IntervalSet(), "leftward": nap.IntervalSet()}

    track_pos = (track_pos - min_pos) / (max_pos - min_pos)
    print(f"\nTrack Position Range after normalization: {track_pos.min():.3f} to {track_pos.max():.3f}")

    # Pre-compute thresholds
    left_end = track_end
    right_end = 1 - left_end
    print(f"Thresholds - Left: {left_end:.3f}, Right: {right_end:.3f}")

    # Find initial run direction using vectorized operations
    left_ix = np.where(track_pos < left_end)[0]
    right_ix = np.where(track_pos > right_end)[0]

    if len(left_ix) == 0 or len(right_ix) == 0:
        return {"rightward": nap.IntervalSet(), "leftward": nap.IntervalSet()}

    rightward_times = []
    leftward_times = []

    # Determine initial direction and starting point
    if left_ix[0] < right_ix[0]:
        next_run = 1  # First run is rightward
        # Find the actual minimum in the left end region
        left_start_region = track_pos[:right_ix[0]]
        nS = np.argmin(left_start_region)
        print(f"\nStarting with rightward run from position {track_pos[nS]:.3f}")
    else:
        next_run = -1  # First run is leftward
        # Find the actual maximum in the right end region
        right_start_region = track_pos[:left_ix[0]]
        nS = np.argmax(right_start_region)
        print(f"\nStarting with leftward run from position {track_pos[nS]:.3f}")

    tot_samples = len(track_pos)
    while nS < tot_samples - 1:
        print(f"\nProcessing run starting at position {track_pos[nS]:.3f}")
        # Use vectorized operations to find next threshold crossing
        if next_run == 1:
            next_ix = np.where(track_pos[nS + 1:] > right_end)[0]
            print(f"Looking for right threshold crossing")
        else:
            next_ix = np.where(track_pos[nS + 1:] < left_end)[0]
            print(f"Looking for left threshold crossing")

        if len(next_ix) == 0:
            print("No more crossings found")
            break

        next_ix = next_ix[0]
        if next_ix == 0:
            nS += 1
            continue

        # Find the actual extremum after crossing threshold
        search_end = min(nS + next_ix + 20, tot_samples)  # Look a bit beyond crossing
        run_range = track_pos[nS + 1:search_end]
        if len(run_range) == 0:
            nS += next_ix
            continue

        # Use vectorized operations to find min/max
        if next_run == 1:
            # For rightward runs, look for minimum after reaching right end
            right_reached_ix = np.where(run_range > right_end)[0]
            if len(right_reached_ix) > 0:
                first_right_ix = right_reached_ix[0]
                remaining_range = run_range[first_right_ix:]
                if len(remaining_range) > 0:
                    run_end = first_right_ix + np.argmin(remaining_range) + nS + 1
                    print(f"Found rightward run end at position {track_pos[run_end]:.3f}")
                else:
                    run_end = nS + next_ix
            else:
                run_end = nS + next_ix
        else:
            # For leftward runs, look for maximum after reaching left end
            left_reached_ix = np.where(run_range < left_end)[0]
            if len(left_reached_ix) > 0:
                first_left_ix = left_reached_ix[0]
                remaining_range = run_range[first_left_ix:]
                if len(remaining_range) > 0:
                    run_end = first_left_ix + np.argmax(remaining_range) + nS + 1
                    print(f"Found leftward run end at position {track_pos[run_end]:.3f}")
                else:
                    run_end = nS + next_ix
            else:
                run_end = nS + next_ix

        # Verify lap reaches both ends using vectorized operations
        lap_segment = track_pos[nS:run_end + 1]
        lap_min = lap_segment.min()
        lap_max = lap_segment.max()

        # Check if the lap covers most of the track
        track_coverage = lap_max - lap_min
        if lap_min <= left_end and lap_max >= right_end and track_coverage >= 0.7:  # Ensure significant coverage
            # Valid lap detected
            if next_run == 1:
                rightward_times.append((timestamps[nS], timestamps[run_end]))
                print(f"Added rightward lap: {track_pos[nS]:.3f} to {track_pos[run_end]:.3f}")
            else:
                leftward_times.append((timestamps[nS], timestamps[run_end]))
                print(f"Added leftward lap: {track_pos[nS]:.3f} to {track_pos[run_end]:.3f}")

        next_run = -next_run
        nS = run_end

    # Create IntervalSets efficiently using numpy arrays
    return {
        "rightward": nap.IntervalSet(
            start=np.array([t[0] for t in rightward_times]),
            end=np.array([t[1] for t in rightward_times])
        ),
        "leftward": nap.IntervalSet(
            start=np.array([t[0] for t in leftward_times]),
            end=np.array([t[1] for t in leftward_times])
        )
    }

def plot_run_times(track_pos_tsd, run_intervals):
    """
    Plot track position vs time with lap periods indicated by shaded backgrounds.

    Parameters:
    track_pos_tsd (nap.Tsd): Pynapple Tsd object containing position data.
    run_intervals (dict): Dictionary containing 'rightward' and 'leftward' nap.IntervalSet objects.
    """
    track_pos_df = track_pos_tsd.as_series().dropna()
    timestamps = track_pos_df.index.values.astype(float)
    track_pos = track_pos_df.values

    # Normalize track position from 0 to 1
    min_pos, max_pos = np.min(track_pos), np.max(track_pos)
    if min_pos != max_pos:
        track_pos = (track_pos - min_pos) / (max_pos - min_pos)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot shaded regions for each run
    for direction, intervals in run_intervals.items():
        color = [0.75, 0.75, 0.75] if direction == "rightward" else [0.9, 0.9, 0.9]
        for start, end in zip(intervals.start, intervals.end):
            ax.axvspan(start, end, color=color, alpha=0.5)

    # Plot track position over time
    ax.plot(timestamps, track_pos, color='k', linewidth=1)

    ax.set_xlim([timestamps[0], timestamps[-1]])
    ax.set_title("Track Position Over Time with Laps")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (normalized)")

    plt.show()

