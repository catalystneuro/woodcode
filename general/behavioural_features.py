import numpy as np
import pandas as pd
import pynapple as nap

def compute_velocity(pos: nap.TsdFrame, window_size: int = None, sampling_rate: float = None) -> nap.Tsd:
    """
    Computes the velocity of an animal based on position data.

    Parameters:
    -----------
    pos : nap.TsdFrame
        A TsdFrame with time as the index and two columns (x, y) for position data.
    window_size : int, optional
        Moving average window size in bins. If None, no smoothing is applied.
    sampling_rate : float, optional
        Sampling rate in Hz (samples per second). If None, the median interval between samples is used.

    Returns:
    --------
    vel : nap.Tsd
        A Tsd containing velocity values indexed by time.
    """

    if not isinstance(pos, nap.TsdFrame):
        raise TypeError("Input must be a pynapple TsdFrame.")

    if pos.shape[1] != 2:
        raise ValueError("Input TsdFrame must have exactly two columns (x, y).")

    # Extract timestamps and position data
    timestamps = pos.index
    pos_values = pos.values  # Convert to NumPy array

    # Determine tracking interval
    if sampling_rate is None:
        tracking_interval = np.median(np.diff(timestamps))  # Compute from timestamps
    else:
        tracking_interval = 1.0 / sampling_rate  # Define explicitly

    # Compute displacement in x and y
    displacement = np.diff(pos_values, axis=0)

    # Compute 2D displacement (Euclidean distance)
    displacement = np.hypot(displacement[:, 0], displacement[:, 1])

    # Compute velocity
    vel = displacement / tracking_interval

    # Convert to pandas Series
    vel_series = pd.Series(vel, index=timestamps[1:])

    # Apply moving average smoothing if window_size is specified
    if window_size is not None:
        vel_series = vel_series.rolling(window=window_size, center=True).mean()

    # Convert back to nap.Tsd
    vel_tsd = nap.Tsd(vel_series.index, vel_series.values)

    return vel_tsd
