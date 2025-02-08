import pynapple as nap

def trim_int_to_tsd(int, tsd):

    tsd_restricted = tsd.restrict(int)
    new_int = nap.IntervalSet(start=tsd_restricted.index[0], end=tsd_restricted.index[-1])

    return new_int



