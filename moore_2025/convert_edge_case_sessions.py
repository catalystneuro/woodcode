"""Convert the edge-case sessions to NWB.

These sessions deviate from the default data layout and exercise the conversion's handling of missing
videos, missing raw OpenEphys output (.dat fallback), an unusual probe (H5), a juvenile recorded with
adult-style alignment, an error epoch, and a clock reset between segments. For the representative
sessions, see `convert_example_sessions.py`.
"""
from pathlib import Path

from moore_2025.session_to_nwb import session_to_nwb


def main():
    """Define paths and convert the edge-case sessions to NWB."""
    juvenile_folder_path = Path("/Volumes/SamsungSSD/CatalystNeuro/Dudchenko/251104_MooreDataset/H3000_Juveniles")
    adult_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/H4800_Adults")
    meta_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/MooreDataset_Metadata.xlsx")
    histology_folder_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/251104_MooreDataset/Histology")
    output_folder_path = Path('/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw')
    juvenile_metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/juvenile_metadata.yaml")
    adult_metadata_file_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/woodcode/moore_2025/adult_metadata.yaml")

    stub_test = False
    save_path = output_folder_path

    # # Edge Case Sessions
    # Example Session without videos
    jv_wt_folder_path = juvenile_folder_path / "WT"
    juvenile_histology_folder_path = histology_folder_path / "H3000"
    folder_name = 'H3001-200202'
    raw_xml_path = jv_wt_folder_path / folder_name / "Raw" / "H3001-200202" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-100.0" / "continuous.xml"
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    timestamps_file_paths = None # Raw Bonsai csv fails to align temporally with the Ephys TTLs --> excluding raw tracking
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_wt_folder_path / folder_name / "Raw" / "H3001-200202"
    save_path = output_folder_path
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
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

    # Example Session without video nor timestamp csv files nor raw OpenEphys output
    adult_wt_folder_path = adult_folder_path / "WT"
    adult_histology_folder_path = histology_folder_path / "H4800"
    folder_name = 'H4823-221108'
    processed_xml_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.xml')  # path to xml file
    raw_xml_path = processed_xml_path
    nrs_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.nrs')  # path to xml file
    mat_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / 'Analysis'
    sleep_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / 'Sleep'
    lfp_file_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.lfp')
    raw_ephys_folder_path = None
    raw_ephys_dat_file_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.dat')  # Raw .dat file for this session since raw OpenEphys folder is missing
    save_path = output_folder_path
    timestamps_file_paths = None
    session_to_nwb(
        folder_name=folder_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        meta_path=meta_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        timestamps_file_paths=timestamps_file_paths,
        lfp_file_path=lfp_file_path,
        raw_ephys_folder_path=raw_ephys_folder_path,
        raw_ephys_dat_file_path=raw_ephys_dat_file_path,
        save_path=save_path,
        metadata_file_path=adult_metadata_file_path,
        histology_folder_path=adult_histology_folder_path,
        stub_test=stub_test,
        is_adult=True,
    )

    # Example Session without raw data
    jv_wt_folder_path = juvenile_folder_path / "WT"
    juvenile_histology_folder_path = histology_folder_path / "H3000"
    folder_name = 'H3023-210812'
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    raw_xml_path = processed_xml_path  # Raw data for this session is missing, so using the Processed XML instead
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_wt_folder_path / folder_name / "Processed" / "BonsaiCaptureALL2021-08-12T19_44_12.avi",
    ]
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Processed" / "BonsaiTracking.csv",
    ]
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_dat_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.dat')  # Raw .dat file for this session since raw OpenEphys folder is missing
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
        raw_ephys_dat_file_path=raw_ephys_dat_file_path,
        save_path=save_path,
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stub_test=stub_test,
        is_adult=False,
    )

    # Example Juvenile Session with Adult temporal alignment and H5 Probe
    jv_wt_folder_path = juvenile_folder_path / "WT"
    juvenile_histology_folder_path = histology_folder_path / "H3000"
    folder_name = 'H3029-230510'
    processed_xml_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    raw_xml_path = processed_xml_path  # Raw XML for this session is missing the SpikeGroup section, so using the Processed XML instead
    nrs_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')  # path to xml file
    mat_path = jv_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = jv_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T12_12_47.avi",
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T12_34_44.avi",
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T14_07_30.avi",
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiVideo2023-05-10T14_31_02.avi",
    ]
    timestamps_file_paths = [
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T12_12_45.csv",
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T12_34_42.csv",
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T14_07_28.csv",
        jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2" / "BonsaiTracking2023-05-10T14_31_01.csv",
    ]
    lfp_file_path = jv_wt_folder_path / folder_name / "Processed" / (folder_name + '.lfp')
    raw_ephys_folder_path = jv_wt_folder_path / folder_name / "Raw" / "day2" / "experiment2"
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
        metadata_file_path=juvenile_metadata_file_path,
        histology_folder_path=juvenile_histology_folder_path,
        stream_name="Acquisition_Board-100.Rhythm Data",
        ttl_stream_name="Acquisition_Board-100.Rhythm Data_ADC",
        stub_test=stub_test,
        is_adult=True,
    )

    # Example Adult Session with error epoch
    adult_wt_folder_path = adult_folder_path / "WT"
    adult_histology_folder_path = histology_folder_path / "H4800"
    folder_name = 'H4830-230406'
    raw_xml_path = adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "Record Node 103" / "experiment1" / "recording1" / "continuous" / "Acquisition_Board-100.Rhythm Data" / "continuous.xml"
    processed_xml_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.xml')  # path to xml file
    nrs_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.nrs')  # path to xml file
    mat_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / 'Analysis'
    sleep_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / 'Sleep'
    video_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T19_28_42.avi",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T21_15_19.avi",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T21_39_18.avi",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiVideo2023-04-06T21_49_57.avi",
    ]
    timestamps_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T19_05_55.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T19_28_41.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T21_15_17.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T21_39_17.csv",
        adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52" / "BonsaiTracking2023-04-06T21_49_56.csv",
    ]
    lfp_file_path = adult_wt_folder_path / folder_name / "Processed" / folder_name / (folder_name + '.lfp')
    raw_ephys_folder_path = adult_wt_folder_path / folder_name / "Raw" / "2023-04-06_19-05-52"
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
        stream_name="Record Node 103#Acquisition_Board-100.Rhythm Data",
        ttl_stream_name="Record Node 103#Acquisition_Board-100.Rhythm Data_ADC",
        stub_test=stub_test,
        is_adult=True,
    )

    # Example session with a clock reset between segments 3 and 4
    adult_wt_folder_path = adult_folder_path / "WT"
    adult_histology_folder_path = histology_folder_path / "H4800"
    folder_name = 'H4815-220814'

    # XMLs for this session are missing a channel, so using a neighbor instead
    raw_xml_path = adult_wt_folder_path / "H4820-221007" / "Raw" / "2022-10-07_18-54-23-day1" / "experiment1" / "recording1" / "continuous" / "Rhythm_FPGA-103.0" / "continuous.xml"
    processed_xml_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.xml')  # path to xml file
    nrs_path = adult_wt_folder_path / folder_name / "Processed" / (folder_name + '.nrs')
    mat_path = adult_wt_folder_path / folder_name / "Processed" / 'Analysis'
    sleep_path = adult_wt_folder_path / folder_name / "Processed" / 'Sleep'
    video_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording1" / "BonsaiVideo2022-08-14T20_19_08.avi",
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording2" / "BonsaiVideo2022-08-14T20_42_41.avi",
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording3" / "BonsaiVideo2022-08-14T22_17_12.avi",
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording4" / "BonsaiVideo2022-08-14T22_48_06.avi",
    ]
    timestamps_file_paths = [
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording1" / "BonsaiTracking2022-08-14T20_19_06.csv",
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording2" / "BonsaiTracking2022-08-14T20_42_39.csv",
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording3" / "BonsaiTracking2022-08-14T22_17_10.csv",
        adult_wt_folder_path / folder_name / "Raw" / "H4815-220814(day1)" / "experiment1" / "recording4" / "BonsaiTracking2022-08-14T22_48_04.csv",
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


if __name__ == "__main__":
    main()
