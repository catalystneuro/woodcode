"""Convert the six representative example sessions to NWB.

Covers both cohorts and genotypes: four juvenile sessions (WT and KO, two days each) and two adult
sessions (WT and KO). For the dataset-wide edge cases, see `convert_edge_case_sessions.py`.
"""
from pathlib import Path

from moore_2025.session_to_nwb import session_to_nwb


def main():
    """Define paths and convert the six example sessions to NWB."""
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

    # Day 2
    folder_name = 'H3022-210806'
    raw_xml_path = jv_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-08-06T11_34_08.avi",
    ]
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "Bonsai testing2021-08-06T11_34_07.csv",
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

    # Example Juvenile KO sessions
    jv_ko_folder_path = juvenile_folder_path / "KO"

    # Day 1
    folder_name = 'H3016-210422'
    raw_xml_path = jv_ko_folder_path / folder_name / "Raw" / "experiment3" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.xml')
    nrs_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_ko_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_ko_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-04-22T18_15_24.avi",
    ]
    timestamps_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "Bonsai testing2021-04-22T18_15_24.csv",
    ]
    lfp_file_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_ko_folder_path / folder_name / "Raw"
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

    # Day 2
    folder_name = 'H3016-210423'
    raw_xml_path = jv_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.xml')
    nrs_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_ko_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_ko_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "BonsaiCaptureALL2021-04-23T14_14_05.avi",
    ]
    timestamps_file_paths = [
        jv_ko_folder_path / folder_name / "Raw" / "Bonsai testing2021-04-23T14_13_55.csv",
    ]
    lfp_file_path = jv_ko_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_ko_folder_path / folder_name / "Raw"
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

    # Example Adult Sessions
    adult_histology_folder_path = histology_folder_path / "H4800"

    # Example Adult WT session
    adult_wt_folder_path = adult_folder_path / "WT"
    folder_name = 'H4813-220728'
    # Note: using the XML from the Raw folder here since the one in Processed is missing one of the channels for shank 2
    processed_xml_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    raw_xml_path = adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-103.0" / "continuous.xml"
    nrs_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = adult_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = adult_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiVideo2022-07-28T18_14_29.avi",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiVideo2022-07-28T18_37_00.avi",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiVideo2022-07-28T20_09_04.avi",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiVideo2022-07-28T20_33_32.avi",
    ]
    timestamps_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiTracking2022-07-28T18_14_27.csv",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiTracking2022-07-28T18_36_58.csv",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiTracking2022-07-28T20_08_59.csv",
        adult_wt_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiTracking2022-07-28T20_33_30.csv",
    ]
    lfp_file_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = adult_wt_folder_path / folder_name / "Raw"
    save_path = output_folder_path
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
        metadata_file_path=adult_metadata_file_path,
        histology_folder_path=adult_histology_folder_path,
        stream_name="Rhythm_FPGA-103.0",
        ttl_stream_name="Rhythm_FPGA-103.0_ADC",
        stub_test=stub_test,
        is_adult=True,
    )

    # Example Adult KO session
    adult_ko_folder_path = adult_folder_path / "KO"
    folder_name = 'H4817-220828'
    # Raw XML for this session is missing one of the channels (channel 38 on shank 1), so using the Processed XML instead
    processed_xml_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    raw_xml_path = processed_xml_path
    nrs_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = adult_ko_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = adult_ko_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiVideo2022-08-28T16_18_06.avi",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiVideo2022-08-28T16_40_58.avi",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiVideo2022-08-28T18_13_23.avi",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiVideo2022-08-28T18_36_14.avi",
    ]
    timestamps_file_paths = [
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording1" / "BonsaiTracking2022-08-28T16_18_05.csv",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording2" / "BonsaiTracking2022-08-28T16_40_56.csv",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording3" / "BonsaiTracking2022-08-28T18_13_20.csv",
        adult_ko_folder_path / folder_name / "Raw" / "experiment1" / "recording4" / "BonsaiTracking2022-08-28T18_36_12.csv",
    ]
    lfp_file_path = adult_ko_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = adult_ko_folder_path / folder_name / "Raw"
    save_path = output_folder_path
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
        metadata_file_path=adult_metadata_file_path,
        histology_folder_path=adult_histology_folder_path,
        stream_name="Rhythm_FPGA-103.0",
        ttl_stream_name="Rhythm_FPGA-103.0_ADC",
        stub_test=stub_test,
        is_adult=True,
    )


if __name__ == "__main__":
    main()
