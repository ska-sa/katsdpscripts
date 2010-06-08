from __future__ import with_statement
import uuid

experiment_id = str(uuid.uuid1())
observer = 'rlord'
description = 'Full 360-degree horizon scan looking for RFI.'
antennas = 'ant1,ant2'
centre_freq = 1800.0  # MHz
dump_rate = 1.0  # Hz
# Scan at elevation 2.2 degrees, centered around 45 degrees azimuth
target = 'azel, 45, 2.2'
scan_duration = 360.0  # seconds

with katuilib.CaptureSession(kat, experiment_id, observer, description, antennas, centre_freq, dump_rate) as session:
    ants = session.ants
    ants.req.drive_strategy('longest-track')
    ants.req.target(target)
    kat.dbe.req.target(target)
    kat.dbe.req.k7w_new_compound_scan(target, 'horizon', 'scan')
    ants.req.scan_sym(180, 0, scan_duration, 'plate-carree')
    print 'Slewing to start of scan'
    ants.req.mode('POINT')
    ants.wait('lock', True, 300)
    print 'Start capturing and scanning'
    kat.dbe.req.capture_start()
    ants.req.mode('SCAN')
    ants.wait('scan_status', 'after', 300)
#   session.fire_noise_diode(diode='coupler', on_duration=10.0, off_duration=10.0)

# Post-processing example
# import scape
# d = scape.DataSet('1234567890.h5')
# scape.plot_xyz(d, 'az', 'amp')
# scape.plot_xyz(d, 'az', 'freq', 'amp')
