"""Insert the six representative example sessions into Spyglass."""

from pathlib import Path

import spyglass.common as sgc
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

from session_to_spyglass import insert_session, print_tables, clear_shared_tables


def main():
    raw_data_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw")

    # Clear existing data for a clean insertion
    clear_shared_tables()

    # Example Juvenile WT
    (sgc.Subject & {"subject_id": "H3022"}).delete()

    # Example Juvenile WT Day 1 Session
    nwbfile_path = raw_data_path / "H3022-210805.nwb"
    table_path = Path("tables_jv_wt_day1.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Juvenile WT Day 2 Session
    nwbfile_path = raw_data_path / "H3022-210806.nwb"
    table_path = Path("tables_jv_wt_day2.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Juvenile KO
    (sgc.Subject & {"subject_id": "H3016"}).delete()

    # Example Juvenile KO Day 1 Session
    nwbfile_path = raw_data_path / "H3016-210422.nwb"
    table_path = Path("tables_jv_ko_day1.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Juvenile KO Day 2 Session
    nwbfile_path = raw_data_path / "H3016-210423.nwb"
    table_path = Path("tables_jv_ko_day2.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Adult WT Session
    nwbfile_path = raw_data_path / "H4813-220728.nwb"
    table_path = Path("tables_ad_wt.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H4813"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)

    # Example Adult KO Session
    nwbfile_path = raw_data_path / "H4817-220828.nwb"
    table_path = Path("tables_ad_ko.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    (sgc.Subject & {"subject_id": "H4817"}).delete()
    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)


if __name__ == "__main__":
    main()
    print("Done!")
