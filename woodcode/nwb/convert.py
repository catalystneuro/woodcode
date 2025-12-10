from pathlib import Path
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries, LFP, TimeSeries
from pynwb.behavior import SpatialSeries, Position, CompassDirection
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common.table import DynamicTable
import scipy.io as spio
from datetime import datetime
import pprint
import numpy as np
import pandas as pd
import warnings
import pynapple as nap
from ndx_franklab_novela import CameraDevice, DataAcqDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup
from pynwb.image import ImageSeries
from neuroconv.utils import calculate_regular_series_rate

def create_nwb_file(metadata, start_time):
    # get info from folder name

    rec_id = metadata['file']['name'].split('-')
    print('Creating NWB file and adding metadata...')

    # calculate animal age
    if metadata['subject']['dob'] is None:
        age_days = 0 # TODO: remove this placeholder once age_metadata has been shared
    else:
        dob_str = str(metadata['subject']['dob'])
        dob = datetime(2000 + int(dob_str[:2]), int(dob_str[2:4]), int(dob_str[4:6]))
        age_days = (start_time.date() - dob.date()).days

    # create an nwb file
    nwbfile = NWBFile(
        session_description=metadata['file']['session_description'],
        identifier=metadata['file']['name'],
        session_start_time=start_time,
        experimenter=metadata['file']['experimenter'],
        experiment_description=metadata['file']['experiment_description'],
        session_id=rec_id[1],
        institution=metadata['file']['institution'],
        keywords=metadata['file']['keywords'].split(', '),
        notes=metadata['file']['notes'],
        protocol=metadata['file']['protocol'],
        related_publications=metadata['file']['related_publications'],
        surgery=metadata['file']['surgery'],
        lab=metadata['file']['lab'],
    )

    # add subject
    nwbfile.subject = Subject(
        age=f"P{age_days}D",
        description=metadata['subject']['description'],
        species='Rattus norvegicus',
        subject_id=rec_id[0],
        genotype=metadata['subject']['genotype'],
        sex=metadata['subject']['sex'],
        strain=metadata['subject']['strain'],
    )

    return nwbfile

def load_nwb_file(file_path, file_name):

    # load NWB file
    file_name = file_name + '.nwb'
    data = nap.load_file(str(file_path / file_name))
    print(data)

    return data


def save_nwb_file(nwbfile, file_path, file_name):
    print('Saving NWB file...')

    # Create the folder path if it doesn't exist
    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)
        print(f'Created directory: {file_path}')

    # Save the NWB file
    with NWBHDF5IO(file_path / (file_name + '.nwb'), 'w') as io:
        io.write(nwbfile)

    print('Done!')


def add_events(nwbfile, events, event_name="events"):
    print('Adding events to NWB file...')

    # Handle case where events is a single IntervalSet
    if isinstance(events, nap.IntervalSet):
        events = {event_name: events}

    # Ensure events is a dictionary
    if not isinstance(events, dict):
        raise TypeError(
            "events must be a dictionary where keys are labels and values are pynapple IntervalSet instances."
        )

    # Ensure all values in events are IntervalSet instances
    if not all(isinstance(interval_set, nap.IntervalSet) for interval_set in events.values()):
        raise TypeError("All values in events must be pynapple IntervalSet instances.")

    # Convert events into a pandas DataFrame
    data = []
    for label, interval_set in events.items():
        df = pd.DataFrame({
            "start_time": interval_set["start"],
            "stop_time": interval_set["end"],
            "label": [label] * len(interval_set)
        })
        data.append(df)

    # Concatenate all event data
    events_df = pd.concat(data, ignore_index=True)

    # Create TimeIntervals from the DataFrame, now with a name
    events_table = TimeIntervals.from_dataframe(events_df, name=event_name)

    # Add to NWB file
    nwbfile.add_time_intervals(events_table)

    return nwbfile



def add_units(nwbfile, xml_data, spikes, waveforms, shank_id):
    print('Adding units to NWB file...')

    # Add extra unit column
    nwbfile.add_unit_column(name="sampling_rate", description="Sampling rate of the raw ephys signal")

    shank_names = list(nwbfile.electrode_groups.keys())
    for ncell in range(len(spikes)):
        group_name = shank_names[shank_id[ncell]]  # Map shank_id to correct name
        nwbfile.add_unit(id=ncell,
                         spike_times=spikes[ncell].index.to_numpy(),
                         waveform_mean=waveforms[ncell].T,
                         sampling_rate=xml_data['dat_sampling_rate'],
                         electrode_group=nwbfile.electrode_groups[group_name])
    return nwbfile


def add_probes(nwbfile, metadata, xmldata, nrsdata, probe_info):
    # to do: add depth info
    """
    Adds probes, electrode groups, and electrodes to the NWB file using Spyglass-compatible types.
    Properly assigns shanks to probes when xmldata['spike_groups']
    is a sequential list instead of a dictionary.
    """

    # get a list of dead channels from the nrs file
    good_channels = nrsdata['channels_shown']

    # Build shank assignments list: each tuple is (probe_id, global_shank_id, probe_location, probe_step, probe_coordinates, probe_reference)
    shank_assignments = []
    global_shank_id = 1 # Global shank ID across all probes
    for probe_metadata in metadata["probe"]:
        probe_id = probe_metadata["id"]
        nshanks = probe_metadata["nshanks"]
        for _ in range(nshanks):
            shank_assignments.append((probe_id, global_shank_id, probe_metadata["location"], probe_metadata["step"], probe_metadata["coordinates"], probe_metadata["reference"]))
            global_shank_id += 1

    # Ensure number of shanks in metadata matches xmldata
    if len(shank_assignments) != len(xmldata["spike_groups"]):
        raise ValueError("Mismatch between shank count in metadata and xmldata['spike_groups']")

    shank_id_to_num_electrodes = {}
    for (_, shank_id, _, _, _, _), electrodes in zip(shank_assignments, xmldata["spike_groups"]):
        shank_id_to_num_electrodes[shank_id] = len(electrodes)

    # Add DataAcqDevice (Spyglass requirement)
    data_acq_device = DataAcqDevice(
        name="data_acquisition_device",
        description=metadata["probe"][0]["data_acquisition_description"],
        system=metadata["probe"][0]["data_acquisition_system"],
        amplifier=metadata["probe"][0]["data_acuisition_amplifier"],
        adc_circuit=metadata["probe"][0]["data_acquisition_adc_circuit"],
    )
    nwbfile.add_device(data_acq_device)

    # Spyglass-required electrode columns
    nwbfile.add_electrode_column(name='probe_shank', description='Shank ID within probe')
    nwbfile.add_electrode_column(name='probe_electrode', description='Electrode ID within shank')
    nwbfile.add_electrode_column(name='bad_channel', description='Boolean indicating if channel is bad')
    nwbfile.add_electrode_column(name='ref_elect_id', description='Reference electrode ID')

    # Build Shank objects with ShanksElectrode objects, organized by probe
    probe_id_to_shanks = {}  # Maps probe_id -> list of Shank objects
    electrode_counter = 0  # Global electrode counter across all shanks and probes
    for probe_id, shank_id, probe_location, probe_step, probe_coordinates, probe_reference in shank_assignments:
        shank_info = probe_info[probe_id][shank_id]
        electrode_coordinates = shank_info['electrode_coordinates']
        num_electrodes = shank_id_to_num_electrodes[shank_id]
        # Initialize probe entry if needed
        if probe_id not in probe_id_to_shanks:
            probe_id_to_shanks[probe_id] = []
        
        # Build ShanksElectrode objects for this shank
        shanks_electrodes = []
        for ielec in range(num_electrodes):
            shanks_electrode = ShanksElectrode(
                name=str(electrode_counter),
                rel_x=electrode_coordinates[ielec][0],
                rel_y=electrode_coordinates[ielec][1],
                rel_z=electrode_coordinates[ielec][2],
            )
            shanks_electrodes.append(shanks_electrode)
            electrode_counter += 1
        
        # Create Shank object and add to probe
        shank = Shank(
            name=str(shank_id),
            shanks_electrodes=shanks_electrodes
        )
        probe_id_to_shanks[probe_id].append(shank)

    # Create Probe devices and add them to nwbfile
    for probe_metadata in metadata["probe"]:
        probe_id = probe_metadata["id"]
        contact_size = probe_info[probe_id]['contact_size']
        probe_name = f"Probe {probe_id}"
        
        probe = Probe(
            name=probe_name,
            description=probe_metadata["description"],
            id=probe_id,
            probe_type=probe_metadata["type"],
            units="um",
            probe_description=probe_metadata["description"],
            contact_side_numbering=True,
            contact_size=contact_size,
            shanks=probe_id_to_shanks[probe_id],
        )
        nwbfile.add_device(probe)

    # Add NwbElectrodeGroup objects for each shank
    for probe_id, shank_id, probe_location, probe_step, probe_coordinates, probe_reference in shank_assignments:
        probe_name = f"Probe {probe_id}"
        probe = nwbfile.devices[probe_name]
        group_name = f"probe{probe_id}_shank{shank_id}"
        electrode_group = NwbElectrodeGroup(
            name=group_name,
            description=f"Electrodes from {group_name}, step: {probe_step}. Targeted (x, y, z) represents (AP, DV, ML) coordinates from Bregma, with Ventral being positive.",
            location=probe_location,
            device=probe,
            targeted_location=probe_location,
            targeted_x=probe_coordinates[0],
            targeted_y=probe_coordinates[1],
            targeted_z=probe_coordinates[2],
            units="mm",
        )
        nwbfile.add_electrode_group(electrode_group)

    # Add Electrodes to the NWBFile
    electrode_counter = 0
    for probe_id, shank_id, probe_location, probe_step, probe_coordinates, probe_reference in shank_assignments:
        shank_info = probe_info[probe_id][shank_id]
        shank_electrode_coordinates = shank_info['electrode_coordinates']
        group_name = f"probe{probe_id}_shank{shank_id}"
        electrode_group = nwbfile.electrode_groups[group_name]
        num_electrodes = shank_id_to_num_electrodes[shank_id]
        for ielec in range(num_electrodes):
            is_bad_channel = electrode_counter not in good_channels
            
            nwbfile.add_electrode(
                rel_x=shank_electrode_coordinates[ielec][0],
                rel_y=shank_electrode_coordinates[ielec][1],
                rel_z=0.0,
                group=electrode_group,
                location=electrode_group.location,
                probe_shank=shank_id,
                probe_electrode=electrode_counter,
                bad_channel=is_bad_channel,
                ref_elect_id=-1, # Spyglass requires this field to be specified as an integer even when none of the probe electrodes served as the original reference.
                reference=probe_reference,
            )
            electrode_counter += 1

    return nwbfile


def add_tracking(nwbfile, pos, ang=None):
    # to do: add units as input
    print('Adding tracking to NWB file...')

    # Create behavior module
    behavior_module = nwbfile.create_processing_module(
        name='behavior',
        description='Behavioral data'
    )

    # Create the spatial series for position
    spatial_series_obj = SpatialSeries(
        name='position',
        description='(x,y) position',
        data=pos.values,
        timestamps=pos.index.to_numpy(),
        reference_frame='', # TODO: add reference frame info once shared
        unit='centimeters'
    )
    position_obj = Position(spatial_series=spatial_series_obj)

    # Add head-direction data only if ang is provided
    if ang is not None:
        data = ang.values[:, np.newaxis]  # Spyglass requires 2D array for all SpatialSeries
        spatial_series_obj = SpatialSeries(
            name='head-direction',
            description='Horizontal angle of the head (yaw)',
            data=data,
            timestamps=ang.index.to_numpy(),
            reference_frame='',
            unit='radians'
        )
        position_obj.add_spatial_series(spatial_series_obj)
        behavior_module.add(position_obj)

    return nwbfile


def add_sleep(nwbfile, sleep_path, folder_name):

    sleep_file = sleep_path / (folder_name + '.SleepState.states.mat')
    emg_file = sleep_path / (folder_name + '.EMGFromLFP.LFP.mat')

    print('Adding sleep stages to NWB file...')

    sleepEpochs = spio.loadmat(sleep_file, simplify_cells=True)
    epWake = np.float32(sleepEpochs['SleepState']['ints']['WAKEstate'])
    epNREM = np.float32(sleepEpochs['SleepState']['ints']['NREMstate'])
    epREM = np.float32(sleepEpochs['SleepState']['ints']['REMstate'])

    # Build a list of dictionaries with row data
    sleep_stage_rows = []
    
    if epREM.size > 0:
        if epREM.ndim == 1:  # in case there is only one interval
            sleep_stage_rows.append({'start_time': epREM[0], 'stop_time': epREM[1], 'tags': ['rem']})
        elif epREM.ndim == 2:
            for nrow in range(len(epREM)):
                sleep_stage_rows.append({'start_time': epREM[nrow, 0], 'stop_time': epREM[nrow, 1], 'tags': ['rem']})

    if epNREM.size > 0:
        if epNREM.ndim == 1:  # in case there is only one interval
            sleep_stage_rows.append({'start_time': epNREM[0], 'stop_time': epNREM[1], 'tags': ['nrem']})
        elif epNREM.ndim == 2:
            for nrow in range(len(epNREM)):
                sleep_stage_rows.append({'start_time': epNREM[nrow, 0], 'stop_time': epNREM[nrow, 1], 'tags': ['nrem']})

    if epWake.size > 0:
        if epWake.ndim == 1:  # in case there is only one interval
            sleep_stage_rows.append({'start_time': epWake[0], 'stop_time': epWake[1], 'tags': ['wake']})
        elif epWake.ndim == 2:
            for nrow in range(len(epWake)):
                sleep_stage_rows.append({'start_time': epWake[nrow, 0], 'stop_time': epWake[nrow, 1], 'tags': ['wake']})

    # Sort rows by start time
    sleep_stage_rows.sort(key=lambda x: x['start_time'])

    # Iterate through the list and add each row to sleep_stages
    sleep_stages = TimeIntervals(name='sleep_stages')
    for row_data in sleep_stage_rows:
        sleep_stages.add_row(**row_data)

    nwbfile.add_time_intervals(sleep_stages)

    print('Adding pseudoEMG to the NWB file...')

    emg = spio.loadmat(emg_file, simplify_cells=True)


    rate = calculate_regular_series_rate(emg['EMGFromLFP']['timestamps'])
    if rate is not None: # If the pseudo-EMG timestamps are perfectly regular, use the more efficient starting time and rate. 
        timestamps = None
        starting_time = float(emg['EMGFromLFP']['timestamps'][0])
    else: # Otherwise, use the timestamps directly.
        timestamps = emg['EMGFromLFP']['timestamps']
        starting_time = None

    emg = TimeSeries(
        name="pseudoEMG",
        description="Pseudo EMG from correlated high-frequency LFP",
        data=emg['EMGFromLFP']['data'],
        unit="a.u.",
        timestamps=timestamps,
        rate=rate,
        starting_time=starting_time,
    )

    # Create an extracellular ephys module or add to the existing one
    if 'ecephys' not in nwbfile.processing:
        ecephys_module = nwbfile.create_processing_module(name='ecephys',
                                                      description='Processed electrophysiological signals'
                                                          )
        ecephys_module.add(emg)
    else:
        nwbfile.processing['ecephys'].add(emg)

    return nwbfile


def add_epochs(nwbfile, epochs, metadata):
    """
    Adds epochs to an NWB file.

    Parameters:
    - nwbfile: NWBFile object
    - epochs: DataFrame containing 'Start' and 'End' times
    - metadata: Dictionary containing epoch metadata

    Returns:
    - Updated NWBFile object
    """

    print('Adding epochs to NWB file...')

    # Extract all tags from metadata
    epoch_tags = {str(i+1): metadata['epoch'].get(str(i+1), None) for i in range(epochs.shape[0])}

    # Check that all epochs have corresponding tags
    if None in epoch_tags.values():
        missing_epochs = [k for k, v in epoch_tags.items() if v is None]
        raise ValueError(f"Missing tags for epochs: {missing_epochs}")

    # Add epochs to NWB file
    for epoch in range(epochs.shape[0]):
        tag = f"{(epoch+1):02d}"  # Spyglass requires 2-digit string epoch numbers
        nwbfile.add_epoch(
            start_time=float(epochs['Start'][epoch]),
            stop_time=float(epochs['End'][epoch]),
            tags=[tag],
        )

    # Add tasks to NWB file
    unique_tasks = set(epoch_tags.values())
    tasks_module = nwbfile.create_processing_module(name="tasks", description="tasks module")
    tasks_metadata = metadata["Task"]
    for task in unique_tasks:
        task_metadata = tasks_metadata[task]
        description = task_metadata["description"]
        environment = task_metadata["environment"]
        camera_id = [task_metadata["camera_id"]]
        task_epochs = [epoch for epoch, tag in epoch_tags.items() if tag == task]
        task_table = DynamicTable(name=task, description=description)
        task_table.add_column(name="task_name", description="Name of the task.")
        task_table.add_column(name="task_description", description="Description of the task.")
        task_table.add_column(name="task_environment", description="The environment the animal was in.")
        task_table.add_column(name="camera_id", description="Camera ID.")
        task_table.add_column(name="task_epochs", description="Task epochs.")
        task_table.add_row(
            task_name=task,
            task_description=description,
            task_environment=environment,
            camera_id=camera_id,
            task_epochs=task_epochs,
        )
        tasks_module.add(task_table)

    return nwbfile



def add_lfp(nwbfile, lfp_path, xml_data, stub_test=False):

    print('Adding LFP to the NWB file...')

    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(len(nwbfile.electrodes))),
        description='all electrodes',
    )

    # get channel numbers in shank order
    chan_order = np.concatenate(xml_data['spike_groups'])

    # lazy load LFP
    lfp_data = nap.load_eeg(filepath=lfp_path, channel=None, n_channels=xml_data['n_channels'], frequency=float(xml_data['eeg_sampling_rate']), precision='int16',
                            bytes_size=2)
    lfp_data = lfp_data[:, chan_order]  # get only probe channels
    if stub_test:
        lfp_data = lfp_data[:100, :]

    # create ElectricalSeries
    lfp_elec_series = ElectricalSeries(
        name='LFP',
        data=H5DataIO(lfp_data, compression=True),  # use this function to compress
        description='Local field potential (downsampled DAT file)',
        electrodes=all_table_region,
        rate=float(xml_data['eeg_sampling_rate'])
    )

    # store ElectricalSeries in an LFP container
    warnings.filterwarnings("ignore",
                            message=".*DynamicTableRegion.*")  # this is to supress a warning here that doesn't seem cause any issues
    lfp = LFP(electrical_series=lfp_elec_series)
    warnings.resetwarnings()

    # Create an extracellular ephys module or add to the existing one
    if 'ecephys' not in nwbfile.processing:
        ecephys_module = nwbfile.create_processing_module(name='ecephys',
                                                      description='Processed electrophysiological signals'
                                                          )
        ecephys_module.add(lfp)
    else:
        nwbfile.processing['ecephys'].add(lfp)


    return nwbfile



def add_misc_tsd(nwbfile, tsd, name='unnamed_series', description='', unit=''):

    """
    Adds a pynapple Tsd or TsdFrame to the 'misc' processing module in an NWB file.

    Parameters:
    - nwbfile: NWBFile object
    - tsd: pynapple Tsd or TsdFrame
    - name: Name of the TimeSeries
    - description: Description of the TimeSeries
    - unit: Unit of the TimeSeries data

    Returns:
    - Updated NWBFile object
    """
    print(f"Adding '{name}' to the NWB file...")

    # grab or create the 'misc' module
    if 'misc' not in nwbfile.processing:
        module = nwbfile.create_processing_module(
            name='misc',
            description='Miscellaneous data'
        )
    else:
        module = nwbfile.processing['misc']


    # make a TimeSeries

    misc_series = TimeSeries(
        name=name,
        data=tsd.values,
        unit=unit,
        timestamps=tsd.index.values
    )

    module.add(misc_series)
    return nwbfile


def collect_nwb_metadata(nwbfile):
    """
    Collects and displays all metadata from an NWB file in a structured format.

    Parameters:
    - nwbfile: An NWBFile object.

    Returns:
    - A dictionary containing all extracted metadata.
    """

    metadata = {}

    # Extract top-level metadata
    metadata["General Info"] = {
        "session_description": nwbfile.session_description,
        "identifier": nwbfile.identifier,
        "session_start_time": nwbfile.session_start_time,
        "experimenter": nwbfile.experimenter,
        "lab": nwbfile.lab,
        "institution": nwbfile.institution,
        "experiment_description": nwbfile.experiment_description,
        "related_publications": nwbfile.related_publications,
        "keywords": nwbfile.keywords,
        "notes": nwbfile.notes,
        "pharmacology": nwbfile.pharmacology,
        "protocol": nwbfile.protocol,
        "slices": nwbfile.slices,
        "source_script": nwbfile.source_script,
        "source_script_file_name": nwbfile.source_script_file_name,
        "data_collection": nwbfile.data_collection,
    }

    # Extract subject metadata
    if nwbfile.subject:
        metadata["Subject"] = {
            "subject_id": nwbfile.subject.subject_id,
            "age": nwbfile.subject.age,
            "description": nwbfile.subject.description,
            "species": nwbfile.subject.species,
            "genotype": nwbfile.subject.genotype,
            "sex": nwbfile.subject.sex,
            "weight": nwbfile.subject.weight,
            "date_of_birth": nwbfile.subject.date_of_birth,
            "strain": getattr(nwbfile.subject, "strain", "N/A"),  # Optional field
        }

    # Extract device metadata
    metadata["Devices"] = {name: device.description for name, device in nwbfile.devices.items()}

    # Extract electrode group metadata
    metadata["Electrode Groups"] = {
        name: {
            "description": group.description,
            "location": group.location,
            "device": group.device.name
        }
        for name, group in nwbfile.electrode_groups.items()
    }

    # Extract electrode table metadata
    if nwbfile.electrodes is not None:
        metadata["Electrodes"] = nwbfile.electrodes.to_dataframe().to_dict()

    # Extract acquisition metadata
    metadata["Acquisition"] = {name: type(obj).__name__ for name, obj in nwbfile.acquisition.items()}

    # Extract processing modules
    metadata["Processing Modules"] = {}
    for module_name, module in nwbfile.processing.items():
        module_data = {}
        for data_name, data in module.data_interfaces.items():
            module_data[data_name] = type(data).__name__
        metadata["Processing Modules"][module_name] = module_data

    # Extract intervals metadata (e.g., epochs)
    metadata["Time Intervals"] = {}
    for interval_name, interval_table in nwbfile.intervals.items():
        if isinstance(interval_table, DynamicTable):
            metadata["Time Intervals"][interval_name] = interval_table.to_dataframe().to_dict()

    # Extract analysis metadata
    metadata["Analysis"] = {name: type(obj).__name__ for name, obj in nwbfile.analysis.items()} if nwbfile.analysis else {}

    # Extract lab metadata
    metadata["Lab Metadata"] = {name: type(obj).__name__ for name, obj in nwbfile.lab_meta_data.items()} if nwbfile.lab_meta_data else {}

    # Print metadata in a structured way
    print("\n=== NWB Metadata ===")
    pprint.pprint(metadata, width=120, compact=False)

    return metadata


def add_video(
    *,
    nwbfile: NWBFile,
    video_file_paths: list[Path],
    all_aligned_video_timestamps: list[np.ndarray],
    metadata: dict,
) -> NWBFile:
    print("Adding video to NWB file...")

    # Add camera device
    camera_device_metadata = metadata["Video"]["CameraDevice"]
    camera_device = CameraDevice(
        name=camera_device_metadata["name"],
        description=camera_device_metadata["description"],
        meters_per_pixel=camera_device_metadata["meters_per_pixel"],
        model=camera_device_metadata["model"],
        lens=camera_device_metadata["lens"],
        camera_name=camera_device_metadata["camera_name"],
    )
    nwbfile.add_device(camera_device)

    # Add image series for each video file
    image_series_metadata = metadata["Video"]["ImageSeries"]
    for meta, timestamps, file_path in zip(image_series_metadata, all_aligned_video_timestamps, video_file_paths):
        image_series = ImageSeries(
            name=meta["name"],
            description=meta["description"],
            external_file=[file_path],
            format="external",
            timestamps=timestamps,
            device=camera_device,
        )
        nwbfile.add_acquisition(image_series)

    return nwbfile

from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
from neuroconv.tools.spikeinterface.spikeinterface import _stub_recording
from neuroconv.utils import calculate_regular_series_rate
import pynwb
from .multi_segment_recording_data_chunk_iterator import MultiSegmentRecordingDataChunkIterator
def add_raw_ephys(nwbfile: NWBFile, folder_path: Path, epochs: pd.DataFrame, xml_data: dict, stream_name: str, stub_test: bool = False) -> NWBFile:
    print("Adding raw ephys to NWB file...")

    chan_order = np.concatenate(xml_data['spike_groups'])

    recording = OpenEphysBinaryRecordingExtractor(folder_path=folder_path, stream_name=stream_name)
    if stub_test:
        recording = _stub_recording(recording)

    # NOTE: spyglass now requires raw electrical series objects to be named, specifically, e-series. 
    eseries_kwargs = dict(name="e-series", description="Acquisition traces for the ElectricalSeries.")

    channel_ids = recording.get_channel_ids()
    region = list(range(len(channel_ids)))
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=region,
        description="electrode_table_region",
    )
    eseries_kwargs.update(electrodes=electrode_table_region)

    if recording.has_scaleable_traces():
        # Spikeinterface gains and offsets are gains and offsets to micro volts.
        # The units of the ElectricalSeries should be volts so we scale correspondingly.
        micro_to_volts_conversion_factor = 1e-6
        channel_gains_to_volts = recording.get_channel_gains() * micro_to_volts_conversion_factor
        channel_offsets_to_volts = recording.get_channel_offsets() * micro_to_volts_conversion_factor

        unique_gains = set(channel_gains_to_volts)
        if len(unique_gains) == 1:
            conversion_to_volts = channel_gains_to_volts[0]
            eseries_kwargs.update(conversion=conversion_to_volts)
        else:
            eseries_kwargs.update(channel_conversion=channel_gains_to_volts)

        unique_offset = set(channel_offsets_to_volts)
        if len(unique_offset) > 1:
            channel_ids = recording.get_channel_ids()
            # This prints a user friendly error where the user is provided with a map from offset to channels
            _report_variable_offset(recording=recording)

        unique_offset = channel_offsets_to_volts[0]
        eseries_kwargs.update(offset=unique_offset)
    else:
        warning_message = (
            "The recording extractor does not have gains and offsets to convert to volts. "
            "That means that correct units are not guaranteed.  \n"
            "Set the correct gains and offsets to the recording extractor before writing to NWB."
        )
        warnings.warn(warning_message, UserWarning, stacklevel=2)

    # Iterator
    segment_indices = list(range(recording.get_num_segments()))
    ephys_data_iterator = MultiSegmentRecordingDataChunkIterator(
        recording=recording,
        segment_indices=segment_indices,
        chan_order=chan_order,
    )
    eseries_kwargs.update(data=ephys_data_iterator)

    timestamps = []
    for i in segment_indices:
        segment_timestamps = recording.get_times(segment_index=i)
        timestamps.append(segment_timestamps)
    timestamps = np.concatenate(timestamps)

    rate = calculate_regular_series_rate(series=timestamps)  # Returns None if it is not regular
    if rate:
        starting_time = float(timestamps[0])
        # Note that we call the sampling frequency again because the estimated rate might be different from the
        # sampling frequency of the recording extractor by some epsilon.
        eseries_kwargs.update(starting_time=starting_time, rate=recording.get_sampling_frequency())
    else:
        eseries_kwargs.update(timestamps=timestamps)

    # Create ElectricalSeries object and add it to nwbfile
    es = pynwb.ecephys.ElectricalSeries(**eseries_kwargs)
    nwbfile.add_acquisition(es)

    return nwbfile


def _report_variable_offset(recording) -> None:
    """
    Helper function to report variable offsets per channel IDs.
    Groups the different available offsets per channel IDs and raises a ValueError.
    """
    channel_offsets = recording.get_channel_offsets()
    channel_ids = recording.get_channel_ids()

    # Group the different offsets per channel IDs
    offset_to_channel_ids = {}
    for offset, channel_id in zip(channel_offsets, channel_ids):
        offset = offset.item() if isinstance(offset, np.generic) else offset
        channel_id = channel_id.item() if isinstance(channel_id, np.generic) else channel_id
        if offset not in offset_to_channel_ids:
            offset_to_channel_ids[offset] = []
        offset_to_channel_ids[offset].append(channel_id)

    # Create a user-friendly message
    message_lines = ["Recording extractors with heterogeneous offsets are not supported."]
    message_lines.append("Multiple offsets were found per channel IDs:")
    for offset, ids in offset_to_channel_ids.items():
        message_lines.append(f"  Offset {offset}: Channel IDs {ids}")
    message = "\n".join(message_lines)

    raise ValueError(message)
