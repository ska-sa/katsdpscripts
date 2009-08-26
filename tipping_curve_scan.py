#!/usr/bin/python
# Perform tipping curve and produce simulated signal data

import ffuilib
import time
import numpy as np

# Build Fringe Finder configuration, as specified in user-facing config file
# The specific configuration is one that runs locally with DBE simulator included
# This connects to all the proxies and devices and queries their commands and sensors
ff = ffuilib.tbuild('cfg-user.ini', 'local_ff_client_sim')

# Clean up any existing experiment
ff.dbe.req_dbe_capture_stop()
ff.ant1.req_mode('STOP')
time.sleep(0.5)

# Elevations at which to evaluate Tsys
elevations = np.arange(85., 0., -25.)
# Do tipping curve at a range of azimuths for each elevation, and select lowest value
azimuths = np.arange(0., 21., 20.)
# Create a sequence of target positions, with azimuth alternately increasing and decreasing
# while elevation decreases from zenith to the horizon
target_az = np.hstack([azimuths[::(-1) ** n] for n in xrange(len(elevations))])
target_el = np.repeat(elevations, len(azimuths))
# Form list of unnamed azel target descriptions - we will have just as many compound scans
targets = ['azel, %s, %s' % (az, el) for az, el in zip(target_az, target_el)]

# Let the data collector know about data file location and format
ff.k7w.req_output_directory('/var/kat/data/')
ff.k7w.req_write_hdf5(1)
# Set the target description string for the first compound scan in the output file
ff.k7w.req_target(targets[0])
# First scan will be a slew to the target - mark it as such before k7w starts
ff.k7w.req_scan_tag('slew')
# Do this BEFORE starting the DBE, otherwise no data will be captured
ff.k7w.req_capture_start()

# Correlator dump rate set to 1 Hz
ff.dbe.req_dbe_dump_rate(1)
# Stream 15 minutes of data (900 dumps) or until stop issued
ff.dbe.req_dbe_packet_count(900)
# Create a new data source labelled "stream", and send data to port 7010 (default k7w data port)
ff.dbe.req_dbe_capture_destination('stream', '127.0.0.1:7010')
# Now start emitting data on stream "stream"
ff.dbe.req_dbe_capture_start('stream')

# Iterate over targets
for compound_scan_id, target in enumerate(targets):
    # Start a new compound scan, which involves a new target description and 'slew' scan label
    ff.k7w.req_target(target)
    ff.k7w.req_scan_tag('slew')
    ff.k7w.req_compound_scan_id(compound_scan_id)

    # Let the antenna slew to the next target and wait for target lock
    # This will be the first scan of the compound scan, labelled 'slew'
    ff.ant1.req_target(target)
    ff.ant1.req_mode('POINT')
    ff.ant1.wait('lock', True, 300)

    # Once we are on the target, start a new scan labelled 'scan'
    ff.k7w.req_scan_id(1, 'scan')
    # Scan duration in seconds
    time.sleep(10)

# Find out which files have been created
files = ff.k7w.req_get_current_files(tuple=True)[1][2]
print 'Data captured to', files

# Stop recording and shut down the experiment
ff.dbe.req_dbe_capture_stop('stream')
ff.k7w.req_capture_stop()
ff.disconnect()
