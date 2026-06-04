"""Insert the edge-case sessions into Spyglass."""

from pathlib import Path

import spyglass.common as sgc
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

from session_to_spyglass import insert_session, print_tables, clear_shared_tables


def main():
    raw_data_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw")

    # Clear existing data for a clean insertion
    clear_shared_tables()

    # Example Session without videos
    nwbfile_path = raw_data_path / "H3001-200202.nwb"
    table_path = Path("tables_jv_wt_no_videos.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H3001"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Session without raw data
    nwbfile_path = raw_data_path / "H3023-210812.nwb"
    table_path = Path("tables_jv_wt_no_raw.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H3023"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Session without video nor timestamp csv files nor raw OpenEphys output
    nwbfile_path = raw_data_path / "H4823-221108.nwb"
    table_path = Path("tables_ad_wt_no_video_no_raw.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H4823"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Juvenile Session with Adult temporal alignment and H5 Probe
    nwbfile_path = raw_data_path / "H3029-230510.nwb"
    table_path = Path("tables_jv_wt_adult_alignment_h5.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H3029"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Adult Session with error epoch
    nwbfile_path = raw_data_path / "H4830-230406.nwb"
    table_path = Path("tables_ad_wt_error_epoch.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H4830"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Session with clock reset between segments
    nwbfile_path = raw_data_path / "H4815-220814.nwb"
    table_path = Path("tables_ad_wt_clock_reset.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H4815"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)


if __name__ == "__main__":
    main()
    print("Done!")
