"""Create a mock NWB file with spyglass-compatible ephys data for testing purposes."""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from ndx_franklab_novela import DataAcqDevice, CameraDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup
from pynwb import NWBHDF5IO
import numpy as np
from pathlib import Path


def main():
    nwbfile = mock_NWBFile(identifier="my_identifier", session_description="my_session_description")

    data_acq_device = DataAcqDevice(
        name="my_data_acq", system="my_system", amplifier="my_amplifier", adc_circuit="my_adc_circuit"
    )
    camera_device = CameraDevice(
        name="Camera 1",
        meters_per_pixel=1.0,
        model="my_model",
        lens="my_lens",
        camera_name="my_camera_name",
    )
    nwbfile.add_device(data_acq_device)
    nwbfile.add_device(camera_device)

    electrode111 = ShanksElectrode(name="1", rel_x=0.0, rel_y=0.0, rel_z=0.0)
    electrode112 = ShanksElectrode(name="2", rel_x=0.0, rel_y=10.0, rel_z=0.0)
    shanks_electrodes11 = [electrode111, electrode112]
    shank11 = Shank(name="1", shanks_electrodes=shanks_electrodes11)
    electrode121 = ShanksElectrode(name="3", rel_x=0.0, rel_y=20.0, rel_z=0.0)
    electrode122 = ShanksElectrode(name="4", rel_x=0.0, rel_y=30.0, rel_z=0.0)
    shanks_electrodes12 = [electrode121, electrode122]
    shank12 = Shank(name="2", shanks_electrodes=shanks_electrodes12)

    probe1 = Probe(
        name="probe1",
        id=0,
        probe_type="probe_type1",
        units="my_units",
        probe_description="my_probe_description",
        contact_side_numbering=False,
        contact_size=1.0,
        shanks=[shank11, shank12],
    )
    nwbfile.add_device(probe1)

    electrode211 = ShanksElectrode(name="5", rel_x=0.0, rel_y=0.0, rel_z=0.0)
    electrode212 = ShanksElectrode(name="6", rel_x=0.0, rel_y=10.0, rel_z=0.0)
    shanks_electrodes21 = [electrode211, electrode212]
    shank21 = Shank(name="1", shanks_electrodes=shanks_electrodes21)
    electrode221 = ShanksElectrode(name="7", rel_x=0.0, rel_y=20.0, rel_z=0.0)
    electrode222 = ShanksElectrode(name="8", rel_x=0.0, rel_y=30.0, rel_z=0.0)
    shanks_electrodes22 = [electrode221, electrode222]
    shank22 = Shank(name="2", shanks_electrodes=shanks_electrodes22)

    probe2 = Probe(
        name="probe2",
        id=1,
        probe_type="probe_type2",
        units="my_units",
        probe_description="my_probe_description",
        contact_side_numbering=False,
        contact_size=1.0,
        shanks=[shank21, shank22],
    )
    nwbfile.add_device(probe2)

    # add electrode groups
    electrode_group1 = NwbElectrodeGroup(
        name="electrode_group1",
        description="my_description",
        location="my_location",
        device=probe1,
        targeted_location="my_targeted_location",
        targeted_x=0.0,
        targeted_y=0.0,
        targeted_z=0.0,
        units="mm",
    )
    nwbfile.add_electrode_group(electrode_group1)
    
    electrode_group2 = NwbElectrodeGroup(
        name="electrode_group2",
        description="my_description",
        location="my_location",
        device=probe2,
        targeted_location="my_targeted_location",
        targeted_x=0.0,
        targeted_y=0.0,
        targeted_z=0.0,
        units="mm",
    )
    nwbfile.add_electrode_group(electrode_group2)
    
    # add electrode columns and electrodes
    extra_cols = [
        "probe_shank",
        "probe_electrode",
        "bad_channel",
        "ref_elect_id",
    ]
    for col in extra_cols:
        nwbfile.add_electrode_column(name=col, description=f"description for {col}")
    
    # add electrodes for probe 1
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group1,
        probe_shank=1,
        probe_electrode=1,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group1,
        probe_shank=1,
        probe_electrode=2,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=10.0,
        z=0.0,
    )
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group1,
        probe_shank=2,
        probe_electrode=3,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=20.0,
        z=0.0,
    )
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group1,
        probe_shank=2,
        probe_electrode=4,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=30.0,
        z=0.0,
    )
    
    # add electrodes for probe 2
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group2,
        probe_shank=1,
        probe_electrode=5,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group2,
        probe_shank=1,
        probe_electrode=6,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=10.0,
        z=0.0,
    )
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group2,
        probe_shank=2,
        probe_electrode=7,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=20.0,
        z=0.0,
    )
    nwbfile.add_electrode(
        location="my_location",
        group=electrode_group2,
        probe_shank=2,
        probe_electrode=8,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=30.0,
        z=0.0,
    )
    
    # add electrical series
    electrodes = nwbfile.electrodes.create_region(name="electrodes", region=np.arange(len(nwbfile.electrodes)), description="electrodes")
    mock_ElectricalSeries(electrodes=electrodes, nwbfile=nwbfile, data=np.ones((10, len(nwbfile.electrodes))))

    # add processing module to make spyglass happy
    nwbfile.create_processing_module(name="behavior", description="dummy behavior module")

    nwbfile_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw/mock_ephys.nwb")
    if nwbfile_path.exists():
        nwbfile_path.unlink()
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)
    print(f"mock ephys NWB file successfully written at {nwbfile_path}")


if __name__ == "__main__":
    main()
