import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def smooth_1d_tuning_curves(
    tuning_curves: pd.DataFrame,
    smooth_factor: float = None,
    circular: bool = True,
    nan_handling: str = "interpolate",
) -> pd.DataFrame:
    """
    Applies Gaussian smoothing to 1D tuning curves with options for circular continuity and NaN handling.

    Parameters
    ----------
    tuning_curves : pd.DataFrame
        A DataFrame where rows represent tuning curve bins and columns represent different units.
    smooth_factor : float, optional
        The standard deviation for Gaussian smoothing. If None, no smoothing is applied.
    circular : bool, optional
        Whether to apply circular smoothing by replicating the tuning curves (default: True).
    nan_handling : str, optional
        How to handle NaNs before smoothing:
        - 'interpolate' (default): Linearly interpolate missing values **after circular extension**.
        - 'zero': Replace NaNs with zeros before smoothing.
        - 'ignore': Leave NaNs untouched (may propagate).

    Returns
    -------
    pd.DataFrame
        A smoothed version of the input tuning curves.
    """

    if smooth_factor is None:
        return tuning_curves.copy()  # Return original DataFrame if no smoothing

    if circular:
        # Extend tuning curves 3 times for circular smoothing
        tmp = np.concatenate((tuning_curves.values, tuning_curves.values, tuning_curves.values), axis=0)

        if nan_handling == "interpolate":
            tmp = pd.DataFrame(tmp).interpolate(method="linear", axis=0).values
        elif nan_handling == "zero":
            tmp = np.nan_to_num(tmp)

        # Apply Gaussian smoothing
        tmp = gaussian_filter1d(tmp, sigma=smooth_factor, axis=0)

        # Trim the concatenated ends
        smoothed_data = tmp[tuning_curves.shape[0]:tuning_curves.shape[0]*2]
    else:
        # Handle NaNs before direct smoothing
        data = tuning_curves.values
        if nan_handling == "interpolate":
            data = tuning_curves.interpolate(method="linear", axis=0).values
        elif nan_handling == "zero":
            data = np.nan_to_num(data)

        # Apply Gaussian smoothing directly
        smoothed_data = gaussian_filter1d(data, sigma=smooth_factor, axis=0)

    return pd.DataFrame(index=tuning_curves.index, data=smoothed_data, columns=tuning_curves.columns)

