import numpy as np

def bayesian_decoder_1d(tuning_curves,
                        binned_spikes,
                        bin_size=None,
                        circular=False,
                        return_posterior=False):
    bins = tuning_curves.index.values
    spikes = binned_spikes.values
    n_time, n_cells = spikes.shape
    n_bins = len(bins)

    # Infer bin size
    if bin_size is None:
        diffs = np.diff(binned_spikes.index.values)
        if np.allclose(diffs, diffs[0]):
            bin_size = diffs[0]
        else:
            bin_size = np.r_[diffs, diffs[-1]]
            if len(bin_size) != n_time:
                raise ValueError("Inferred bin_size does not match number of time bins.")

    # tuning curves as (n_bins, n_cells)
    tc = tuning_curves.values
    log_tc = np.log(tc + 1e-12)

    if np.ndim(bin_size) == 0:   # scalar
        # log-likelihood: (n_time, n_bins)
        log_p = -bin_size * tc.sum(1)[None, :] \
                + spikes @ log_tc.T
    else:  # vector of bin sizes (n_time,)
        # First term depends on time
        term1 = -np.outer(bin_size, tc.sum(1))        # (n_time, n_bins)
        # Second term: spikes (n_time, n_cells) × log_tc.T (n_cells, n_bins)
        term2 = spikes @ log_tc.T
        log_p = term1 + term2

    # flat prior
    log_p += -np.log(n_bins)

    # normalize with softmax trick
    log_p -= log_p.max(1, keepdims=True)
    p = np.exp(log_p)
    p /= p.sum(1, keepdims=True)

    # decode
    if circular:
        decoded = np.angle(np.exp(1j * bins) @ p.T)
    else:
        decoded = (bins @ p.T)

    if return_posterior:
        return decoded, p
    else:
        return decoded