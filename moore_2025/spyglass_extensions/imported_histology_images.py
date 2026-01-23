import datajoint as dj

from spyglass.common.common_nwbfile import Nwbfile  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("imported_histology_images")


@schema
class ImportedHistologyImages(SpyglassMixin, dj.Imported):
    definition = """
    -> Session                      # the session to which this PseudoEMG belongs
    image_name: varchar(40)      # name of the histology image
    ---
    histology_image_object_id: varchar(40)      # object ID of a histology image for loading from the NWB file
    """

    _nwb_table = Nwbfile

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        images = nwbf.acquisition["histology_images"].images
        for image_name, image_object in images.items():
            insert_key = {
                "nwb_file_name": nwb_file_name,
                "image_name": image_name,
                "histology_image_object_id": image_object.object_id,
            }
            self.insert1(insert_key, allow_direct_insert=True)