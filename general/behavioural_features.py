import numpy as np
import pandas as pd
import pynapple as nap

def compute_velocity(pos, window_size: int = None, sampling_rate: float = None):
    """
    Computes the velocity of an animal based on position data.

    Parameters:
    -----------
    pos : pd.DataFrame or nap.Tsd
        Position data with time as the index.
        - If a DataFrame, it should have two columns (x, y).
        - If a Tsd, it should contain 2D position values as an array-like structure.
    window_size : int, optional
        Moving average window size in bins. If None, no smoothing is applied.
    sampling_rate : float, optional
        Sampling rate in Hz (samples per second). If None, the median interval between samples is used.

    Returns:
    --------
    vel : nap.Tsd
        A time series of velocity values.
    """

    # Convert nap.Tsd to pandas DataFrame if needed
    if isinstance(pos, nap.Tsd):
        timestamps = pos.index
        pos_values = np.array(pos.values)  # Ensure it's a NumPy array
        if pos_values.shape[1] != 2:
            raise ValueError("Input nap.Tsd must contain 2D position values (x, y).")
        pos = pd.DataFrame(pos_values, index=timestamps, columns=["x", "y"])

    elif not isinstance(pos, pd.DataFrame):
        raise TypeError("Input must be either a pandas DataFrame or a pynapple Tsd.")

    # Determine tracking interval
    if sampling_rate is None:
        tracking_interval = np.median(np.diff(pos.index))  # Compute from timestamps
    else:
        tracking_interval = 1.0 / sampling_rate  # Define explicitly

    # Compute displacement in x and y
    displacement = np.diff(pos.to_numpy(), axis=0)

    # Compute 2D displacement (Euclidean distance)
    displacement = np.hypot(displacement[:, 0], displacement[:, 1])

    # Compute velocity
    vel = displacement / tracking_interval

    # Convert to pandas Series
    vel_series = pd.Series(vel, index=pos.index[1:])

    # Apply moving average smoothing if window_size is specified
    if window_size is not None:
        vel_series = vel_series.rolling(window=window_size, center=True).mean()

    # Convert back to nap.Tsd
    vel_tsd = nap.Tsd(vel_series.index, vel_series.values)

    return vel_tsd
