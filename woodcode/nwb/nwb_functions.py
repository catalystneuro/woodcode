from pathlib import Path

import pynapple as nap
from pynwb import NWBHDF5IO, NWBFile
from pynwb.behavior import SpatialSeries, Position, CompassDirection
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.image import ImageSeries
import numpy as np
import cv2  # OpenCV for reading video frames
from hdmf.data_utils import DataChunkIterator


def create_nwb_file(metadata, start_time):
    # get info from folder name
    rec_id = metadata['recording'].split('-')
    print('Creating NWB file and adding metadata...')

    # create an nwb file
    nwbfile = NWBFile(
        session_description=metadata['session_description'],
        experiment_description=metadata['experiment_description'],
        identifier=rec_id[0],
        session_start_time=start_time,
        session_id=rec_id[1],
        experimenter='Duszkiewicz, Adrian J.',
        lab='Wood/Dudchenko lab',
        institution='University of Edinburgh',
        virus='',
        related_publications='',
        keywords=['head-direction', 'postsubiculum', 'extracellular', 'freely-moving', 'electrophysiology'])

    # add subject
    age_weeks = metadata['age_weeks']
    nwbfile.subject = Subject(age=f'P{age_weeks}W',
                              description=metadata['line'],
                              species='Rattus norvegicus',
                              subject_id=rec_id[0],
                              genotype=metadata['genotype'],
                              sex=metadata['sex'])

    return nwbfile

def load_nwb_file(datapath, foldername):

    # load NWB file
    filepath = datapath / foldername
    filename = foldername + '.nwb'
    filepath = filepath / filename

    data = nap.load_file(str(filepath))
    print(data)

    return data

def save_nwb_file(nwbfile,datapath, foldername):
    print('Saving NWB file...')
    with NWBHDF5IO(datapath / foldername / (foldername + '.nwb'), 'w') as io:
        io.write(nwbfile)
    print('Done!')

def add_events(nwbfile, events):
    print('Adding events to NWB file...')

  # Create a TimeIntervals table
    events_table = TimeIntervals(name="events")
    events_table.add_column(name="event_type", description="Type of event")  # Add a column for event type

    for timestamp, state in events.items():
        events_table.add_row(start_time=timestamp, stop_time=timestamp+0.001, event_type=state) # stop time doesn't matter

    nwbfile.add_time_intervals(events_table)

    return nwbfile


def add_units(nwbfile, spikes, waveforms, shank_id):

    print('Adding units to NWB file...')
    for ncell in range(len(spikes)):
        nwbfile.add_unit(id=ncell,
                         spike_times=spikes[ncell],
                         waveform_mean=waveforms[ncell].T,
                         electrode_group=nwbfile.electrode_groups[f'shank{shank_id[ncell]}'])

    return nwbfile


def add_probe(nwbfile, metadata, **kwargs):
    '''
    n_shanks: number of shanks
    n_channels: number of channels per shank
    step: spacing between electrodes (in um)
    electrode_counter: number of electrodes added so far
    '''
    print('Adding probes to NWB file...')
    n_shanks = kwargs.get('n_shanks')
    n_channels = kwargs.get('n_channels')
    step = kwargs.get('step')
    electrode_counter = kwargs.get('electrodes_so_far')

    # add columns to the electrode table
    nwbfile.add_electrode_column(name='label', description='label of electrode')
    nwbfile.add_electrode_column(name='is_faulty', description='Boolean column to indicate faulty electrodes')

    # create probe device
    device = nwbfile.create_device(
        name=metadata['probe'],
        description=metadata['probe_description']
    )

    # now add electrodes
    for ishank in range(n_shanks):
        #create an electrode group for this shank
        electrode_group = nwbfile.create_electrode_group(
            name='shank{}'.format(ishank),
            description='electrode group for shank{}'.format(ishank),
            device=device,
            location=metadata['probe_target'],
        )

        for ielec in range(n_channels):
            elec_depth = step * (n_channels - ielec-1)
            nwbfile.add_electrode(
                x=0., y=float(elec_depth), z=0.,  # add electrode position
                location=metadata['probe_target'],
                filtering='none',
                is_faulty=False,  # TODO
                group=electrode_group,
                label='shank{}elec{}'.format(ishank,ielec)
            )
            electrode_counter += 1

    # define table region RAW DAT FILE and LFP will refer to (all electrodes)
    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(electrode_counter)),
        description='all electrodes',
    )

    return nwbfile


def add_tracking(nwbfile, pos, ang):

    print('Adding tracking to NWB file...')
    # create behaviour module
    behavior_module = nwbfile.create_processing_module(
        name='behavior',
        description='Tracking data acquired with Bonsai'
    )

    # create the spatial series for position
    spatial_series_obj = SpatialSeries(
        name='position',
        description='(x,y) position',
        data=pos.values,
        timestamps=pos.index.to_numpy(),
        reference_frame='',  # TODO
        unit='centimeters'
    )
    position_obj = Position(spatial_series=spatial_series_obj)

    # create the spatial series for head-direction
    spatial_series_obj = SpatialSeries(
        name='head-direction',
        description='Horizontal angle of the head (yaw)',
        data=ang.values,
        timestamps=ang.index.to_numpy(),
        reference_frame='',  # TODO
        unit='radians'
    )
    direction_obj = CompassDirection(spatial_series=spatial_series_obj)

    # update behaviour module
    behavior_module.add(position_obj)
    behavior_module.add(direction_obj)

    return nwbfile


def add_epochs(nwbfile, epochs, metadata):
    print('Adding epochs to NWB file...')
    for epoch in range(epochs.shape[0]):
        nwbfile.add_epoch(start_time=float(epochs['Start'][epoch]), stop_time=float(epochs['End'][epoch]), tags=metadata[f'epoch_{epoch+1}'])

    return nwbfile








