#!/usr/bin/python
# Raster scan across a simulated DBE target producing scan data for signal displays and loading into scape (non CaptureSession version).
#
# This script is basically a cut and paste from some of the underlying functionality provided by
# katuilib's observe.py in order to show how scripts can also be created using the lower-level
# functionality if more explicit control is needed. Usually, it is much simpler to use observe.py
# (see simscan.py script, for example, which does much the same as this script).

import katuilib
import time
import uuid
import numpy as np
from optparse import OptionParser

usage = "usage: %prog [options]"
parser = OptionParser(usage=usage)

parser.add_option('-i', '--ini_file', dest='ini_file', type="string", metavar='INI', help='Telescope configuration ' +
                  'file to use in conf directory (default reuses existing connection, or falls back to cfg-local.ini)')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", metavar='SELECTED',
                  help='Selected configuration to use (default reuses existing connection, or falls back to local_ff)')
parser.add_option('-a', '--ants', dest='ants', type="string", default="ant1,ant2", metavar='ANTS',
                  help='Comma separated list of antennas to include in scan (default="%default")')
(opts, args) = parser.parse_args()

experiment_id = str(uuid.uuid1()) # generate a unique experiment ID
observer = 'nobody'
description = 'Sim target raster scan example (verbose)'
centre_freq=1800.0
dump_rate=1.0
target = 'Takreem,azel,20,30' # if simulation target is changed, also update in line kat.dbe.req.dbe_test_target()
num_scans=3
scan_duration=20.0
scan_extent=4.0
scan_spacing=0.5
scan_in_azimuth=True
drive_strategy='shortest-slew'
label='raster'
record_slews=True

# Try to build the given KAT configuration (which might be None, in which case try to reuse latest active connection)
# This connects to all the proxies and devices and queries their commands and sensors
try:
    kat = katuilib.tbuild(opts.ini_file, opts.selected_config)
# Fall back to *local* configuration to prevent inadvertent use of the real hardware
except ValueError:
    kat = katuilib.tbuild('cfg-local.ini', 'local_ff')
print "\nUsing KAT connection with configuration: %s\n" % (kat.get_config(),)

# create an array using the specified antennas
ants = katuilib.Array("ants", [getattr(kat, ant.strip()) for ant in opts.ants.split(',')])

########################################################################################
########## The following lines correspond to CaptureSession.__init__ function ##########
########################################################################################

# Start with a clean state, by stopping the DBE
kat.dbe.req.capture_stop()

# Set centre frequency in RFE stage 7
kat.rfe7.req.rfe7_lo1_frequency(4200.0 + centre_freq, 'MHz')
effective_lo_freq = (centre_freq - 200.0) * 1e6

# Set data output directory (typically on ff-dc machine)
kat.dbe.req.k7w_output_directory("/var/kat/data")
# Enable output to HDF5 file (takes effect on capture_start only), and set basic experimental info
kat.dbe.req.k7w_write_hdf5(1)
kat.dbe.req.k7w_experiment_info(experiment_id, observer, description)

# The DBE proxy needs to know the dump period (in ms) as well as the effective LO freq,
# which is used for fringe stopping (eventually). This sets the delay model and other
# correlator parameters, such as the dump rate, and instructs the correlator to pass
# its data to the k7writer daemon (set via configuration)
kat.dbe.req.capture_setup(1000.0 / dump_rate, effective_lo_freq)

print "New data capturing session"
print "--------------------------"
print "Experiment ID =", experiment_id
print "Observer =", observer
print "Description ='%s'" % description
print "RF centre frequency = %g MHz, dump rate = %g Hz, keep slews = %s" % \
      (centre_freq, dump_rate, record_slews)

# If the DBE is simulated, it will have position update commands
if hasattr(kat.dbe.req, 'dbe_pointing_az') and hasattr(kat.dbe.req, 'dbe_pointing_el'):
    first_ant = ants.devs[0]
    # Tell the DBE simulator where the first antenna is so that it can generate target flux at the right time
    # The minimum time between position updates is just a little less than the standard (az, el) sensor period
    first_ant.sensor.pos_actual_scan_azim.register_listener(kat.dbe.req.dbe_pointing_az, 0.4)
    first_ant.sensor.pos_actual_scan_elev.register_listener(kat.dbe.req.dbe_pointing_el, 0.4)
    print "DBE simulator receives position updates from antenna '%s'" % (first_ant.name,)

###########################################################################################
########## The following lines correspond to CaptureSession.raster_scan function ##########
###########################################################################################

# Set the drive strategy for how antenna moves between targets
ants.req.drive_strategy(drive_strategy)
# Set the antenna target
ants.req.target(target)
# Provide target to the DBE proxy, which will use it as delay-tracking center
kat.dbe.req.target(target)

########## SIMULATOR-SPECIFIC LINES ##########
# Comment out the following line to run this script on the real hardware
# Tell the DBE simulator to make a test target at specified az and el
kat.dbe.req.dbe_test_target(20, 30, 100)
##############################################

# Create new CompoundScan group in HDF5 file, which automatically also creates the first Scan group
kat.dbe.req.k7w_new_compound_scan(target, label, 'slew' if record_slews else 'scan')

# Create start positions of each scan, based on scan parameters
scan_steps = np.arange(-(num_scans // 2), num_scans // 2 + 1)
scanning_coord = (scan_extent / 2.0) * (-1) ** scan_steps
stepping_coord = scan_spacing * scan_steps
# These minus signs ensure that the first scan always starts at the top left of target
scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)

# The next block will actually start the recording - wrap this in a try...finally to clean up recording afterwards
try:
    # Iterate through the scans across the target
    for scan_count, scan in enumerate(scan_starts):

        print "Slewing to start of scan %d of %d on target '%s'" % (scan_count + 1, len(scan_starts), target)
        if record_slews:
            # Create a new Scan group in HDF5 file, with 'slew' label (not necessary the first time)
            if scan_count > 0:
                kat.dbe.req.k7w_new_scan('slew')
            # If we haven't yet, start recording data from the correlator (which creates the file)
            if kat.dbe.sensor.capturing.get_value() == '0':
                kat.dbe.req.capture_start()
        # Move each antenna to the start position of the next scan
        if scan_in_azimuth:
            ants.req.scan_asym(scan[0], scan[1], -scan[0], scan[1], scan_duration)
        else:
            ants.req.scan_asym(scan[0], scan[1], scan[0], -scan[1], scan_duration)
        ants.req.mode('POINT')
        # Wait until they are all in position (with 5 minute timeout)
        ants.wait('lock', True, 300)

        print "Starting scan %d of %d on target '%s'" % (scan_count + 1, len(scan_starts), target)
        if record_slews or (scan_count > 0):
            # Start a new Scan group in the HDF5 file, labelled as a proper 'scan'
            kat.dbe.req.k7w_new_scan('scan')
        # Unpause HDF5 file output (or create data file and start recording if not done yet)
        kat.dbe.req.k7w_write_hdf5(1)
        if not record_slews and (kat.dbe.sensor.capturing.get_value() == '0'):
            kat.dbe.req.capture_start()
        # Start scanning the antennas
        ants.req.mode('SCAN')
        # Wait until they are all finished scanning (with 5 minute timeout)
        ants.wait('scan_status', 'after', 300)
        # If slews are not to be recorded, pause the file output again directly after the scan
        if not record_slews:
            kat.dbe.req.k7w_write_hdf5(0)

########################################################################################
########## The following lines correspond to CaptureSession.shutdown function ##########
########################################################################################

    # Obtain the name of the file currently being written to
    reply = kat.dbe.req.k7w_get_current_file()
    outfile = reply[1].replace('writing', 'unaugmented') if reply.succeeded else '<unknown file>'
    print 'Scans complete, data captured to %s' % (outfile,)

finally:
    # Stop the DBE data flow (this indirectly stops k7writer via a stop packet, which then closes the HDF5 file)
    kat.dbe.req.capture_stop()
    print 'Ended data capturing session with experiment ID', experiment_id
