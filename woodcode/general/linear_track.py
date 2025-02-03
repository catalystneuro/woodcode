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

    # First, find all local minima and maxima using gradient
    gradient = np.gradient(track_pos)
    potential_min_idx = []
    potential_max_idx = []

    # Find zero crossings in gradient for minima and maxima
    for i in range(1, len(gradient)):
        if gradient[i - 1] < 0 and gradient[i] >= 0:  # Local minimum
            potential_min_idx.append(i - 1)
        elif gradient[i - 1] > 0 and gradient[i] <= 0:  # Local maximum
            potential_max_idx.append(i - 1)

    # Filter extrema by position thresholds
    valid_min_idx = [idx for idx in potential_min_idx if track_pos[idx] <= left_end]
    valid_max_idx = [idx for idx in potential_max_idx if track_pos[idx] >= right_end]

    if not valid_min_idx or not valid_max_idx:
        return {"rightward": nap.IntervalSet(), "leftward": nap.IntervalSet()}

    rightward_times = []
    leftward_times = []

    print("\nLap Start and End Points:")
    print("-" * 40)

    # Process all minima and maxima in temporal order to find laps
    all_extrema = [(idx, 'min') for idx in valid_min_idx] + [(idx, 'max') for idx in valid_max_idx]
    all_extrema.sort(key=lambda x: x[0])

    current_idx = 0
    while current_idx < len(all_extrema) - 1:
        start_idx, start_type = all_extrema[current_idx]
        end_idx, end_type = all_extrema[current_idx + 1]

        # Check if this pair forms a valid lap
        segment = track_pos[start_idx:end_idx + 1]
        lap_min = segment.min()
        lap_max = segment.max()

        valid_lap = False
        # For rightward laps
        if start_type == 'min' and end_type == 'min' and lap_max >= right_end:
            rightward_times.append((timestamps[start_idx], timestamps[end_idx]))
            print(f"Rightward lap: Start={track_pos[start_idx]:.3f}, End={track_pos[end_idx]:.3f}")
            print(f"  - Lap min: {lap_min:.3f}, Lap max: {lap_max:.3f}")
            valid_lap = True
        # For leftward laps
        elif start_type == 'max' and end_type == 'max' and lap_min <= left_end:
            leftward_times.append((timestamps[start_idx], timestamps[end_idx]))
            print(f"Leftward lap: Start={track_pos[start_idx]:.3f}, End={track_pos[end_idx]:.3f}")
            print(f"  - Lap min: {lap_min:.3f}, Lap max: {lap_max:.3f}")
            valid_lap = True

        current_idx += 1 if not valid_lap else 1

    print("\nExtremum points found:")
    print("Minima:", [f"{track_pos[idx]:.3f}" for idx in valid_min_idx])
    print("Maxima:", [f"{track_pos[idx]:.3f}" for idx in valid_max_idx])

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

