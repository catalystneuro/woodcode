import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau


def smooth_1d_tuning_curves(tuning_curves,
                            smooth_factor: float = None,
                            circular: bool = False,
                            nan_handling: str = "interpolate"
                            ) -> np.ndarray:
    """
    Applies Gaussian smoothing to 1D tuning curves with options for circular continuity and NaN handling.

    Parameters
    ----------
    tuning_curves : pd.DataFrame, pd.Series, or np.ndarray
        1D or 2D input where rows represent tuning curve bins and columns represent different units.
    smooth_factor : float, optional
        The standard deviation for Gaussian smoothing. If None, no smoothing is applied.
    circular : bool, optional
        Whether to apply circular smoothing by replicating the tuning curves (default: True).
    nan_handling : str, optional
        How to handle NaNs before smoothing:
        - 'interpolate' (default): Linearly interpolate missing values **before circular extension**.
        - 'zero': Replace NaNs with zeros before smoothing.
        - 'ignore': Leave NaNs untouched (may propagate).

    Returns
    -------
    np.ndarray
        Smoothed tuning curves as a NumPy array.
    """

    # Convert input to a NumPy array (if not already)
    if isinstance(tuning_curves, (pd.DataFrame, pd.Series)):
        tuning_curves = tuning_curves.to_numpy()

    if not isinstance(tuning_curves, np.ndarray):
        raise TypeError("tuning_curves must be a Pandas DataFrame, Series, or a NumPy ndarray.")

    if smooth_factor is None:
        return tuning_curves.copy()  # No smoothing, return original array

    # Ensure 2D array for consistency (handles both 1D and 2D cases)
    tuning_curves = np.atleast_2d(tuning_curves)

    # Handle NaNs efficiently
    if nan_handling == "interpolate":
        # Find NaN indices
        nan_mask = np.isnan(tuning_curves)
        if np.any(nan_mask):
            x = np.arange(tuning_curves.shape[0])
            for i in range(tuning_curves.shape[1]):  # Loop over columns
                col = tuning_curves[:, i]
                if np.any(np.isnan(col)):
                    valid_mask = ~nan_mask[:, i]
                    tuning_curves[:, i] = np.interp(x, x[valid_mask], col[valid_mask])
    elif nan_handling == "zero":
        np.nan_to_num(tuning_curves, copy=False)  # In-place replacement

    # Circular smoothing
    if circular:
        # Extend for circular smoothing
        n_bins = tuning_curves.shape[0]
        extended = np.concatenate((tuning_curves, tuning_curves, tuning_curves), axis=0)

        # Apply Gaussian filter
        smoothed_data = gaussian_filter1d(extended, sigma=smooth_factor, axis=0, mode="wrap")

        # Trim back to original size
        smoothed_data = smoothed_data[n_bins:n_bins*2]
    else:
        # Apply Gaussian smoothing directly
        smoothed_data = gaussian_filter1d(tuning_curves, sigma=smooth_factor, axis=0, mode="nearest")

    return smoothed_data  # Always returns a NumPy array


def compute_1d_tuning_correlation(data1, data2, method: str = "pearson", circular: bool = False) -> np.ndarray:
    """
    Compute column-wise correlation between corresponding columns of two inputs.
    Optionally, perform circular shifting of data2 before computing correlations.

    Parameters:
    data1 (pd.DataFrame, pd.Series, or np.ndarray): First input (features x bins).
    data2 (pd.DataFrame, pd.Series, or np.ndarray): Second input (features x bins).
    method (str): Correlation method, one of 'pearson', 'spearman', or 'kendall'. Default is 'pearson'.
    circular (bool): Whether to compute correlations with circular shifts of data2. Default is False.

    Returns:
    np.ndarray: If circular=False, a (1, n_features) array with correlations.
                If circular=True, an (n_bins, n_features) array where each row corresponds to a shift.
    """

    # Convert Pandas inputs to NumPy arrays
    if isinstance(data1, (pd.DataFrame, pd.Series)):
        data1 = data1.to_numpy()
    if isinstance(data2, (pd.DataFrame, pd.Series)):
        data2 = data2.to_numpy()

    # Ensure inputs are 2D arrays
    data1 = np.atleast_2d(data1)
    data2 = np.atleast_2d(data2)

    if data1.shape != data2.shape:
        raise ValueError("Both inputs must have the same shape.")

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Method must be one of 'pearson', 'spearman', or 'kendall'.")

    n_bins, n_cols = data1.shape

    if method == "spearman":
        # Rank transform for Spearman correlation
        data1 = np.apply_along_axis(lambda x: x.argsort().argsort(), axis=0, arr=data1)
        data2 = np.apply_along_axis(lambda x: x.argsort().argsort(), axis=0, arr=data2)

    if not circular:
        # Standard correlation
        if method in {"pearson", "spearman"}:
            corrs = np.array([np.corrcoef(data1[:, i], data2[:, i])[0, 1] for i in range(n_cols)])
        else:  # Kendall
            corrs = np.array([kendalltau(data1[:, i], data2[:, i])[0] for i in range(n_cols)])

        return corrs.reshape(1, -1)  # Shape: (1, n_features)

    else:
        # Circular mode: Compute correlations for each shift
        shift_correlations = np.zeros((n_bins, n_cols))

        for shift in range(n_bins):
            rolled_data2 = np.roll(data2, shift=shift, axis=0)  # Fast circular shift
            if method in {"pearson", "spearman"}:
                corrs = np.array([np.corrcoef(data1[:, i], rolled_data2[:, i])[0, 1] for i in range(n_cols)])
            else:  # Kendall
                corrs = np.array([kendalltau(data1[:, i], rolled_data2[:, i])[0] for i in range(n_cols)])

            shift_correlations[shift, :] = corrs  # Store correlations for each shift

        return shift_correlations  # Shape: (n_bins, n_features)

