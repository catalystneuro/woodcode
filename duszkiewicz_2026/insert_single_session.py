"""Insert the Duszkiewicz example session (H6813-240605) into Spyglass."""

from pathlib import Path

import spyglass.common as sgc
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

from duszkiewicz_2026.session_to_spyglass import insert_session, print_tables, clear_shared_tables


def main():
    raw_data_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw")

    # Clear shared probe/camera/device/task records for a clean insertion
    clear_shared_tables()

    # Clear any prior insertion of this subject/session
    (sgc.Subject & {"subject_id": "H6813"}).delete()

    nwbfile_path = raw_data_path / "H6813-240605.nwb"
    table_path = Path("tables_duszkiewicz_H6813-240605.txt")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()

    insert_session(nwbfile_path, rollback_on_fail=True, raise_err=True)
    print_tables(nwbfile_path=nwbfile_path, table_path=table_path)


if __name__ == "__main__":
    main()
    print("Done!")
