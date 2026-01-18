import datajoint as dj
import numpy as np
import pynwb

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import (
    AnalysisNwbfile,
    Nwbfile,
)  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup  # noqa: F401
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_nwb_file,
    get_valid_intervals,
)

schema = dj.schema("imported_pseudo_emg")


@schema
class ImportedPseudoEMG(SpyglassMixin, dj.Imported):
    definition = """
    -> Session                      # the session to which this PseudoEMG belongs
    -> IntervalList                 # the original set of times to be filtered
    pseudoemg_object_id: varchar(40)      # object ID of a pseudoemg electrical series for loading from the NWB file
    ---
    pseudoemg_sampling_rate: float        # the sampling rate, in samples/sec
    """

    _nwb_table = Nwbfile

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        pseudo_emg_time_series = nwbf.processing["ecephys"]["pseudoEMG"]
        timestamps = pseudo_emg_time_series.get_timestamps()

        # estimate the sampling rate or read in if available
        sampling_rate = pseudo_emg_time_series.rate or estimate_sampling_rate(timestamps[: int(1e6)])

        # create a new interval list for the valid times
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"imported pseudoemg valid times",
            "valid_times": get_valid_intervals(timestamps, sampling_rate),
            "pipeline": "imported_pseudoemg",
        }
        IntervalList().insert1(interval_key)

        # build key to insert into ImportedPseudoEMG
        insert_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_key["interval_list_name"],
            "pseudoemg_object_id": pseudo_emg_time_series.object_id,
            "pseudoemg_sampling_rate": sampling_rate,
        }
        self.insert1(insert_key, allow_direct_insert=True)