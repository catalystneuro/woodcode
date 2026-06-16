"""Convert the Duszkiewicz example session (H6813-240605) to NWB.

Run from the repo root with:
    python -m duszkiewicz_2025.convert_single_session
"""
from pathlib import Path

from duszkiewicz_2025.session_to_nwb import session_to_nwb


def main():
    """Define paths and convert the example Duszkiewicz session to NWB."""
    dataset_path = Path("/Volumes/T7/CatalystNeuro/Dudchenko/260612_DuszkiewiczDataset")
    session_path = dataset_path / "H6800_Syngap" / "HET" / "H6813-240605"
    output_folder_path = Path("/Users/pauladkisson/Documents/CatalystNeuro/DudchenkoConv/Spyglass/raw")
    metadata_file_path = Path(__file__).parent / "duszkiewicz_metadata.yaml"

    folder_name = "H6813-240605"
    metadata_lookup_name = "H6813_240605"
    stub_test = False

    record_node_path = session_path / "Raw" / "2024-06-05_22-04-41" / "Record Node 101"
    experiment_names = ["experiment1", "experiment2"]
    stream_name = "Record Node 101#Acquisition_Board-100.Rhythm Data"
    ttl_stream_name = "Record Node 101#Acquisition_Board-100.Rhythm Data_ADC"

    raw_xml_path = (
        record_node_path / "experiment1" / "recording1" / "continuous"
        / "Acquisition_Board-100.Rhythm Data" / "continuous.xml"
    )
    processed_xml_path = session_path / "Processed" / (folder_name + ".xml")
    nrs_path = session_path / "Processed" / (folder_name + ".nrs")
    lfp_file_path = session_path / "Processed" / (folder_name + ".lfp")
    meta_path = dataset_path / "DuszkiewiczDataset_Metadata.xlsx"
    mat_path = session_path / "Processed" / "Analysis"
    sleep_path = session_path / "Processed" / "Sleep"
    cue_epochs_path = mat_path / "CueEpochs.mat"
    ttl_folder_path = (
        record_node_path / "experiment2" / "recording1" / "events"
        / "Acquisition_Board-100.Rhythm Data" / "TTL"
    )

    video_file_paths = [
        next((record_node_path / experiment_name / "recording1").glob("BonsaiVideo*.avi"))
        for experiment_name in experiment_names
    ]
    timestamps_file_paths = [
        next((record_node_path / experiment_name / "recording1").glob("BonsaiTracking*.csv"))
        for experiment_name in experiment_names
    ]

    session_to_nwb(
        folder_name=folder_name,
        metadata_lookup_name=metadata_lookup_name,
        record_node_path=record_node_path,
        experiment_names=experiment_names,
        stream_name=stream_name,
        ttl_stream_name=ttl_stream_name,
        raw_xml_path=raw_xml_path,
        processed_xml_path=processed_xml_path,
        nrs_path=nrs_path,
        lfp_file_path=lfp_file_path,
        meta_path=meta_path,
        metadata_file_path=metadata_file_path,
        mat_path=mat_path,
        sleep_path=sleep_path,
        cue_epochs_path=cue_epochs_path,
        ttl_folder_path=ttl_folder_path,
        video_file_paths=video_file_paths,
        timestamps_file_paths=timestamps_file_paths,
        save_path=output_folder_path,
        stub_test=stub_test,
    )


if __name__ == "__main__":
    main()
