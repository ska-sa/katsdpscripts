#!/usr/bin/python
# Perform Tsys scan at zenith and produce simulated signal data

import ffuilib
import time

ff = ffuilib.tbuild('cfg-user.ini', 'local_ff_client_sim')

# Clean up any existing experiment
ff.dbe.req_dbe_capture_stop()
ff.ant1.req_mode('STOP')
time.sleep(0.5)

# Perform Tsys measurement at zenith
target = 'Zenith, azel, 0, 90'

# Let the data collector know the target, data file format and location
ff.k7w.req_target(target)
ff.k7w.req_output_directory('/var/kat/data/')
ff.k7w.req_write_hdf5(1)
# First scan will be a slew to the target - mark it as such before k7w starts
ff.k7w.req_scan_tag('slew')
ff.k7w.req_capture_start()

# Stream 10 minutes of data or until stop issued
ff.dbe.req_dbe_packet_count(900)
# Correlator dump rate set to 1 Hz
ff.dbe.req_dbe_dump_rate(1)
# Create a new data source labelled "stream", and send data to port 7010 (default k7w data port)
ff.dbe.req_dbe_capture_destination('stream', '127.0.0.1:7010')
# Start emitting data on stream "stream"
ff.dbe.req_dbe_capture_start('stream')

# Let the antenna slew to the target and wait for target lock
# This will be the first scan of the compound scan, labelled 'slew'
ff.ant1.req_target(target)
ff.ant1.req_mode('POINT')
ff.ant1.wait('lock', True, 300)

# Once we are on the target, start a new scan labelled 'scan'
ff.k7w.req_scan_id(1, 'scan')
# Scan duration
time.sleep(60)

files = ff.k7w.req_get_current_files(tuple=True)[1][2]
print 'Data captured to', files

# Stop recording and shut down the experiment
ff.dbe.req_dbe_capture_stop('stream')
ff.k7w.req_capture_stop()
ff.disconnect()
