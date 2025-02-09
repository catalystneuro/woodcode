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

    # Find all left and right threshold crossings
    left_crosses = np.where(track_pos < left_end)[0]
    right_crosses = np.where(track_pos > right_end)[0]

    if len(left_crosses) == 0 or len(right_crosses) == 0:
        return {"rightward": nap.IntervalSet(), "leftward": nap.IntervalSet()}

    rightward_times = []
    leftward_times = []

    # Determine initial direction and starting point
    if left_crosses[0] < right_crosses[0]:
        next_run = 1  # First run is rightward
        # Find the first minimum (start of rightward run)
        search_end = right_crosses[0]
        nS = np.argmin(track_pos[:search_end]) if search_end > 0 else 0
    else:
        next_run = -1  # First run is leftward
        # Find the first maximum (start of leftward run)
        search_end = left_crosses[0]
        nS = np.argmax(track_pos[:search_end]) if search_end > 0 else 0

    tot_samples = len(track_pos)
    lap_start_idx = nS

    while nS < tot_samples - 1:
        # Find next crossing
        remaining_pos = track_pos[nS + 1:]
        if next_run == 1:
            # For rightward runs, look for right threshold crossing
            next_crosses = np.where(remaining_pos > right_end)[0]
        else:
            # For leftward runs, look for left threshold crossing
            next_crosses = np.where(remaining_pos < left_end)[0]

        if len(next_crosses) == 0:
            break

        next_cross = next_crosses[0] + nS + 1

        # Find the start of the next lap
        search_end = min(next_cross + 100, tot_samples)

        if next_run == 1:
            # Current run is rightward, look for maximum (start of next leftward run)
            next_lap_start = np.argmax(track_pos[next_cross:search_end]) + next_cross
            # Find the minimum between current start and next start - this is the true end of current lap
            end_segment = track_pos[next_cross:next_lap_start + 1]
            if len(end_segment) > 0:
                run_end = next_cross + np.argmin(end_segment)
            else:
                run_end = next_cross
        else:
            # Current run is leftward, look for minimum (start of next rightward run)
            next_lap_start = np.argmin(track_pos[next_cross:search_end]) + next_cross
            # Find the maximum between current start and next start - this is the true end of current lap
            end_segment = track_pos[next_cross:next_lap_start + 1]
            if len(end_segment) > 0:
                run_end = next_cross + np.argmax(end_segment)
            else:
                run_end = next_cross

        # Verify lap reaches both ends by checking the whole segment
        lap_segment = track_pos[lap_start_idx:run_end + 1]
        if len(lap_segment) > 0:
            lap_min = lap_segment.min()
            lap_max = lap_segment.max()

            # Check if it's a valid lap
            if lap_min <= left_end and lap_max >= right_end:
                if next_run == 1:
                    rightward_times.append((timestamps[lap_start_idx], timestamps[run_end]))
                else:
                    leftward_times.append((timestamps[lap_start_idx], timestamps[run_end]))

        # Update for next iteration
        lap_start_idx = next_lap_start
        nS = next_lap_start
        next_run = -next_run

    return {
        "direction_1": nap.IntervalSet(
            start=np.array([t[0] for t in rightward_times]),
            end=np.array([t[1] for t in rightward_times])
        ),
        "direction_2": nap.IntervalSet(
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
        color = [0.75, 0.75, 0.75] if direction == "direction_1" else [0.9, 0.9, 0.9]
        for start, end in zip(intervals.start, intervals.end):
            ax.axvspan(start, end, color=color, alpha=0.5)

    # Plot track position over time
    ax.plot(timestamps, track_pos, color='k', linewidth=1)

    ax.set_xlim([timestamps[0], timestamps[-1]])
    ax.set_title("Track Position Over Time with Laps")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (normalized)")

    plt.show()


