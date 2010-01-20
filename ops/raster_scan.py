#!/usr/bin/python
# Raster scan across a target, producing scan data for signal displays and loading into scape

import ffuilib
import time
from optparse import OptionParser
import numpy as np

# Dump rate, in Hz
dump_rate = 1.0
# RF centre frequency, in GHz
centre_freq = 1.8

# Scan parameters
# Number of scans (odd is better, as this will scan directly over the source)
num_scans = 3
# Scanning in azimuth along the same elevation?
azimuth_scan = True
# Start and end of each scan is offset by this amount (in degrees) from the target
# along the coordinate that is scanned (e.g. in azimuth)
scan_offset = 2.0
# The separation between each consecutive scan along the coordinate that is not scanned (e.g. elevation)
scan_spacing = 0.5
# The duration of each scan, in seconds
scan_duration = 20.0
# Set the drive strategy for how antenna moves between targets
# (options are: "longest-track", the default, or "shortest-slew")
drive_strategy = "shortest-slew"

# Parse command-line options that allow the defaults to be overridden
# Default FF configuration is *local*, to prevent inadvertent use of the real hardware
parser = OptionParser(usage="usage: %prog [options]")
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                  help='Telescope configuration file to use in conf directory (default="%default")')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                  help='Selected configuration to use (default="%default")')
parser.add_option('-a', '--ants', dest='ants', type="string", default="ant1,ant2", metavar='ANTS',
                  help='Comma-separated list of antennas to include in scan (default="%default")')
parser.add_option('-t', '--tgt', dest='tgt', type="string", default="Takreem,azel,20,30", metavar='TGT',
                  help='Target to scan, as description string (default="%default")')
(opts, args) = parser.parse_args()
tgt = opts.tgt

#######################################################################################################################
############## No need to edit the stuff below, unless you are feeling adventurous or the system changed ##############
#######################################################################################################################


########## Connect to Fringe Finder hardware ##########


# Build Fringe Finder configuration, as specified in user-facing config file
# This connects to all the proxies and devices and queries their commands and sensors
ff = ffuilib.tbuild(opts.ini_file, opts.selected_config)


########## Set up list of antennas and scans ##########


# Create a list of the specified antenna devices, and complain if they are not found
try:
    ants = [eval("ff." + ant_x) for ant_x in opts.ants.split(",")]
except AttributeError:
    raise ValueError("Antenna '%s' not found" % ant_x)
# Create list of baselines, based on the selected antennas
# (1 = ant1-ant1 autocorr, 2 = ant1-ant2 cross-corr, 3 = ant2-ant2 autocorr)
# This determines which HDF5 files are created
if ff.ant1 in ants and ff.ant2 in ants:
    baselines = [1, 2, 3]
elif ff.ant1 in ants and ff.ant2 not in ants:
    baselines = [1]
elif ff.ant1 not in ants and ff.ant2 in ants:
    baselines = [3]
else:
    baselines = []

########## Initialise hardware and start data capturing ##########


# Start with a clean state, by stopping the DBE
ff.dbe.req.capture_stop()

# Initialise antennas
for ant_x in ants:
    # Set the drive strategy for how antenna moves between targets
    ant_x.req.drive_strategy(drive_strategy)
    # Set the antenna target
    ant_x.req.target(tgt)

# Set centre frequency in RFE stage 7
ff.rfe7.req.rfe7_lo1_frequency(4.2 + centre_freq, 'GHz')
effective_lo_freq = (centre_freq - 0.2) * 1e9

# Set data output directory (typically on ff-dc machine)
ff.dbe.req.k7w_output_directory("/var/kat/data")
# Provide target to k7_writer, which will put it in data file
ff.dbe.req.k7w_target(tgt)
# Tell k7_writer to write the selected baselines to HDF5 files
ff.dbe.req.k7w_baseline_mask(*baselines)
ff.dbe.req.k7w_write_hdf5(1)
# Set the compound scan ID (only one) and initial scan ID - the first scan will be a slew to start of first proper scan
ff.dbe.req.k7w_compound_scan_id(0)
ff.dbe.req.k7w_scan_id(0, 'slew')

# Provide target to the DBE proxy, which will use it as delay-tracking center
ff.dbe.req.target(tgt)
# This is a precaution to prevent bad timestamps from the correlator
ff.dbe.req.dbe_sync_now()
# The DBE proxy needs to know the dump rate as well as the effective LO freq, which is used for fringe stopping (eventually)
ff.dbe.req.capture_setup(1000.0 / dump_rate, effective_lo_freq)
# Start recording data from the correlator!
ff.dbe.req.capture_start()


########## Perform scans with antennas and indicate scan structure for data files ##########


# Create start positions of each scan, based on scan parameters
scan_steps = np.arange(-(num_scans // 2), num_scans // 2 + 1)
scanning_coord = scan_offset * (-1) ** scan_steps
non_scanning_coord = scan_spacing * scan_steps
# These minus signs ensure that the first scan always starts at the top left of target
scan_starts = zip(scanning_coord, -non_scanning_coord) if azimuth_scan else zip(non_scanning_coord, -scanning_coord)

# Iterate through the scans across the target
for scan_count, scan in enumerate(scan_starts):

    print "Slewing to start of scan %d of %d" % (scan_count + 1, len(scan_starts))
    # Set the new scan ID - this will create a new Scan group in the HDF5 file (except on the first scan)
    ff.dbe.req.k7w_scan_id(2*scan_count, 'slew')
    # Send each antenna to the start position of the next scan
    for ant_x in ants:
        if azimuth_scan:
            ant_x.req.scan_asym(scan[0], scan[1], -scan[0], scan[1], scan_duration)
        else:
            ant_x.req.scan_asym(scan[0], scan[1], scan[0], -scan[1], scan_duration)
        ant_x.req.mode("POINT")
    # Wait until they are all in position (with 5 minute timeout)
    for ant_x in ants:
        ant_x.wait("lock", True, 300)

    print "Starting scan %d of %d" % (scan_count + 1, len(scan_starts))
    # Start a new Scan group in the HDF5 file, this time labelled as a proper 'scan'
    ff.dbe.req.k7w_scan_id(2*scan_count + 1, "scan")
    # Start scanning the antennas
    for ant_x in ants:
        ant_x.req.mode("SCAN")
    # Wait until they are all finished scanning (with 5 minute timeout)
    for ant_x in ants:
        ant_x.wait("scan_status", "after", 300)


########## Stop recording and close connection to hardware ##########


# Obtain the names of the files currently being written to
files = ff.dbe.req.k7w_get_current_files(tuple=True)[1][2]
print "Scans complete, data captured to", files

# Stop the data capture and close the Fringe Finder connections
ff.dbe.req.capture_stop()
ff.disconnect()


# now augment the hdf5 files with metadata (pointing info etc):
# scp ffuser@ff-dc:/var/kat/data/xxx.yyy.h5 .
# augment.py -i <ini_file> -s <selected_config> -d . -f xxx.yyy.h5

# load into scape from within python:
#   import scape
#   import pylab as pl
#   d = scape.DataSet("[xxx.h5]") # load data from file into dataset
#   print d
#   scape.plot_compound_scan_on_target(d.compscans[1])
#   pl.show()
#   d = d.select(labelkeep="scan") # get rid of the slew data from the dataset
#   print d
#   print d.compscans[0]
#   scape.plot_compound_scan_on_target(d.compscans[0])
