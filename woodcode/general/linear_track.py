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

    # Pre-compute thresholds
    left_end = track_end
    right_end = 1 - left_end

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
        nS = left_ix[0]
    else:
        next_run = -1  # First run is leftward
        nS = right_ix[0]

    tot_samples = len(track_pos)
    while nS < tot_samples - 1:
        # Use vectorized operations to find next threshold crossing
        if next_run == 1:
            next_ix = np.where(track_pos[nS + 1:] > right_end)[0]
        else:
            next_ix = np.where(track_pos[nS + 1:] < left_end)[0]

        if len(next_ix) == 0:
            # No more crossings, check if final segment is a valid lap
            run_end = tot_samples - 1
        else:
            next_ix = next_ix[0]
            if next_ix == 0:
                nS += 1
                continue

            # Find the actual end of the run (min/max point)
            run_range = track_pos[nS + 1:nS + next_ix + 1]
            if len(run_range) == 0:
                nS += next_ix
                continue

            # Use vectorized operations to find min/max
            if next_run == 1:
                run_end = np.argmin(run_range) + nS + 1
            else:
                run_end = np.argmax(run_range) + nS + 1

        # Verify lap reaches both ends using vectorized operations
        lap_segment = track_pos[nS:run_end + 1]
        lap_min = lap_segment.min()
        lap_max = lap_segment.max()

        if lap_min <= left_end and lap_max >= right_end:
            # Valid lap detected
            if next_run == 1:
                rightward_times.append((timestamps[nS], timestamps[run_end]))
            else:
                leftward_times.append((timestamps[nS], timestamps[run_end]))

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

