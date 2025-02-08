import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trim_int_to_tsd(int, tsd):

    tsd_restricted = tsd.restrict(int)
    new_int = nap.IntervalSet(start=tsd_restricted.index[0], end=tsd_restricted.index[-1])

    return new_int

def get_waveform_features(nwbfile, plot_result=False):
    """
    Extracts waveform features from an NWB file, selecting the waveform with the highest amplitude
    and computing trough-to-peak duration for each cell without upsampling.

    Parameters:
    - nwbfile: NWB file object containing spike waveforms.
    - plot_result (bool, optional): Whether to plot the result. Default is False.

    Returns:
    - waveform_features (pd.DataFrame): DataFrame containing waveform features, including trough-to-peak durations.
    """

    # Get waveform and sampling rate info from the NWB file
    waveforms = nwbfile.nwb.units['waveform_mean'].data[:]
    sampling_rate = nwbfile.nwb.units['sampling_rate'].data[0]

    # Unpack shape dynamically
    n_cells, n_samples, n_channels = waveforms.shape

    # Compute sum of squared values for each channel
    sum_squared = np.sum(waveforms ** 2, axis=1)  # (n_cells, n_channels)

    # Find the index of the channel with the maximum sum for each cell
    max_channel_indices = np.argmax(sum_squared, axis=1)  # (n_cells,)

    # Extract the max waveform efficiently
    max_waveforms = np.take_along_axis(waveforms, max_channel_indices[:, None, None], axis=2).squeeze(axis=2)

    # Generate time vector in milliseconds
    t = np.linspace(0, (n_samples - 1) / sampling_rate * 1000, n_samples)

    # Find trough (minimum value index) in a vectorized manner
    trough_indices = np.argmin(max_waveforms, axis=1)  # (n_cells,)

    # Find peak after the trough
    peak_indices = np.array([
        trough_idx + np.argmax(max_waveforms[cell_idx, trough_idx:])
        for cell_idx, trough_idx in enumerate(trough_indices)
    ])

    # Compute trough-to-peak durations in milliseconds
    trough_to_peak_durations = t[peak_indices] - t[trough_indices]

    # Create a DataFrame for the results
    waveform_features = pd.DataFrame({
        'Cell_Index': np.arange(n_cells),
        'Max_Channel_Index': max_channel_indices,
        'Trough_to_Peak': trough_to_peak_durations
    })

    if plot_result:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot all mean waveforms individually, normalized
        for i in range(n_cells):
            min_val = np.min(max_waveforms[i])
            normalized_waveform = max_waveforms[i] / abs(min_val)  # Normalize so min is 1

            axs[0].plot(t, normalized_waveform, label=f"Cell {i}")

        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Normalized Amplitude")
        axs[0].set_title("Normalized Mean Waveforms for Each Cell")

        # Plot histogram of trough-to-peak durations
        axs[1].hist(trough_to_peak_durations, bins=20, color="blue", alpha=0.7, edgecolor="black")
        axs[1].set_xlabel("Trough-to-Peak Duration (ms)")
        axs[1].set_ylabel("Count")
        axs[1].set_title("Distribution of Trough-to-Peak Durations")

        plt.tight_layout()
        plt.show()

    return waveform_features
