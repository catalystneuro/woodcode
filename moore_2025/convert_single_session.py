"""Convert the first example session to NWB.
"""
from pathlib import Path

from moore_2025.session_to_nwb import session_to_nwb


def main():
    """Define paths and convert the first example session to NWB."""
    juvenile_folder_path = Path("/Volumes/SamsungSSD/CatalystNeuro/Dudchenko/251104_MooreDataset/H3000_Juveniles")
    adult_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/H4800_Adults")
    meta_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/MooreDataset_Metadata.xlsx")
    histology_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/Histology")
    output_folder_path = Path('/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw')
    juvenile_metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/juvenile_metadata.yaml")
    adult_metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/adult_metadata.yaml")

    stub_test = False
    save_path = output_folder_path

    # Example Juvenile Sessions
    juvenile_histology_folder_path = histology_folder_path / "H3000"

    # Example Juvenile WT sessions
    jv_wt_folder_path = juvenile_folder_path / "WT"

    # Day 1
    folder_name = 'H3022-210805'
    raw_xml_path = jv_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-08-05T17_06_24.avi",
    ]
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "Bonsai testing2021-08-05T17_06_23.csv",
    ]
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_wt_folder_path / folder_name / "Raw"

    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stream_name="Rhythm_FPGA-100.0",
        ttl_stream_name="Rhythm_FPGA-100.0_ADC",
        stub_test=stub_test,
        is_adult=False,
    )


if __name__ == "__main__":
    main()
