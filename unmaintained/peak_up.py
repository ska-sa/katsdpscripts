#!/usr/bin/python
# Perform quick raster scan on point source and use it to fit a beam to derive a pointing offset.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import optparse
import time

import numpy as np
import matplotlib.pyplot as plt

import katuilib
from katuilib.observe import ant_array

import katpoint
import scape
import scikits.fitting as fit

class InputError(Exception):
    """Error in option or argument specified on the command line."""
    pass

# Parse command-line options that allow the defaults to be overridden
parser = optparse.OptionParser(usage="%prog [options] <target>", description="Peak up on point source.")
# Generic options
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", metavar='INI', help='Telescope configuration ' +
                  'file to use in conf directory (default reuses existing connection, or falls back to cfg-local.ini)')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", metavar='SELECTED',
                  help='Selected configuration to use (default reuses existing connection, or falls back to local_ff)')
parser.add_option('-a', '--ants', dest='ants', type="string", metavar='ANTS',
                  help="Comma-separated list of antennas to include in scan (e.g. 'ant1,ant2')," +
                       " or 'all' for all antennas (**required** - safety reasons)")
parser.add_option('-f', '--centre_freq', dest='centre_freq', type="float", default=1822.0,
                  help='Centre frequency, in MHz (default="%default")')
(opts, args) = parser.parse_args()

if len(args) == 0:
    raise InputError("Please specify at least one target argument (via name, e.g. 'Cygnus A' or description, e.g. 'azel, 20, 30')")
# Various non-optional options...
if opts.ants is None:
    raise InputError('Please specify the antennas to use via -a option (yes, this is a non-optional option...)')

# Try to build the given KAT configuration (which might be None, in which case try to reuse latest active connection)
# This connects to all the proxies and devices and queries their commands and sensors
try:
    kat = katuilib.tbuild(opts.ini_file, opts.selected_config)
# Fall back to *local* configuration to prevent inadvertent use of the real hardware
except ValueError:
    kat = katuilib.tbuild('cfg-local.ini', 'local_ff')
print "\nUsing KAT connection with configuration: %s\n" % (kat.get_config(),)

# Look up target name in catalogue, and keep target description string as is
if args[0].find(',') < 0:
    # With no comma in the target string, assume it's the name of a target to be looked up in the standard catalogue
    target = kat.sources[args[0]]
    if target is None:
        raise InputError("Unknown source '%s'" % (args[0],))
else:
    # Assume the argument is a target description string
    target = katpoint.Target(args[0])

##### SETUP CAPTURE SESSION #####

# Dump rate in Hz
dump_rate = 1.0

# Create antenna array
ants = ant_array(kat, opts.ants)
# Don't create an HDF5 file
kat.dbe.req.k7w_write_hdf5(0)
# Start with a clean state, by stopping the DBE
kat.dbe.req.capture_stop()
# Set centre frequency in RFE stage 7
kat.rfe7.req.rfe7_lo1_frequency(4200.0 + opts.centre_freq, 'MHz')
effective_lo_freq = (opts.centre_freq - 200.0) * 1e6
# The DBE proxy needs to know the dump period (in ms) as well as the effective LO freq,
# which is used for fringe stopping (eventually). This sets the delay model and other
# correlator parameters, such as the dump rate, and instructs the correlator to pass
# its data to the k7writer daemon (set via configuration)
kat.dbe.req.capture_setup(1000.0 / dump_rate, effective_lo_freq)

print "New data capturing session"
print "--------------------------"
print "RF centre frequency = %g MHz, dump rate = %g Hz" % (opts.centre_freq, dump_rate)

# If the DBE is simulated, it will have position update commands
if hasattr(kat.dbe.req, 'dbe_pointing_az') and hasattr(kat.dbe.req, 'dbe_pointing_el'):
    first_ant = ants.devs[0]
    # Tell the DBE simulator where the first antenna is so that it can generate target flux at the right time
    # The minimum time between position updates is just a little less than the standard (az, el) sensor period
    first_ant.sensor.pos_actual_scan_azim.register_listener(kat.dbe.req.dbe_pointing_az, 0.4)
    first_ant.sensor.pos_actual_scan_elev.register_listener(kat.dbe.req.dbe_pointing_el, 0.4)
    print "DBE simulator receives position updates from antenna '%s'" % (first_ant.name,)

##### SETUP SCAN PARAMETERS #####

# Set scan parameters
num_scans = 3
scan_duration = 30.0
scan_extent = 6.0
scan_spacing = 0.5
scan_in_azimuth = True
drive_strategy = 'shortest-slew'

# Set the drive strategy for how antenna moves between targets
ants.req.drive_strategy(drive_strategy)
# Set the antenna target
ants.req.target(target)
# Provide target to the DBE proxy, which will use it as delay-tracking center
kat.dbe.req.target(target)

# Create start positions of each scan, based on scan parameters
scan_steps = np.arange(-(num_scans // 2), num_scans // 2 + 1)
scanning_coord = (scan_extent / 2.0) * (-1) ** scan_steps
stepping_coord = scan_spacing * scan_steps
# These minus signs ensure that the first scan always starts at the top left of target
scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)

##### PERFORM SCAN #####

# If we haven't yet, start recording data from the correlator
if kat.dbe.sensor.capturing.get_value() == '0':
    kat.dbe.req.capture_start()
# Start signal display session
if kat.dh.sd is None:
    kat.dh.start_sdisp()

# Iterate through the scans across the target
for scan_count, scan in enumerate(scan_starts):
    print "Slewing to start of scan %d of %d on target '%s'" % (scan_count + 1, len(scan_starts), target.name)
    # Move each antenna to the start position of the next scan
    if scan_in_azimuth:
        ants.req.scan_asym(scan[0], scan[1], -scan[0], scan[1], scan_duration)
    else:
        ants.req.scan_asym(scan[0], scan[1], scan[0], -scan[1], scan_duration)
    ants.req.mode('POINT')
    # Wait until they are all in position (with 5 minute timeout)
    ants.wait('lock', True, 300)
    # Mark the start of the first scan
    if scan_count == 0:
        start_time = time.time()
    print "Starting scan %d of %d on target '%s'" % (scan_count + 1, len(scan_starts), target.name)
    # Start scanning the antennas
    ants.req.mode('SCAN')
    # Wait until they are all finished scanning (with 5 minute timeout)
    ants.wait('scan_status', 'after', 300)

end_time = time.time()
kat.dbe.req.capture_stop()

##### EXTRACT DATA #####

# Extract data from first antenna only for now
first_ant = ants.devs[0]
ant_idx = int(first_ant.name[3:])
# Extract az-el coordinates
az = first_ant.sensor.pos_actual_scan_azim.get_stored_history(start_time=start_time, end_time=end_time, select=False)
el = first_ant.sensor.pos_actual_scan_elev.get_stored_history(start_time=start_time, end_time=end_time, select=False)
# Extract autocorrelation data (pick the first usable polarisation)
try:
    corr_prod = kat.dh.sd.cpref.user_to_id((ant_idx, ant_idx, 'HH'))
except KeyError:
    try:
        corr_prod = kat.dh.sd.cpref.user_to_id((ant_idx, ant_idx, 'VV'))
    except KeyError:
        raise KeyError("Could not find usable autocorrelation data for antenna '%s'" % (first_ant.name,))
timestamps, power = kat.dh.sd.select_data(product=corr_prod, start_time=start_time, end_time=end_time, avg_axis=1,
                                          start_channel=100, stop_channel=400, include_ts=True)

##### FIT BEAM AND BASELINE #####

# Query KAT antenna for antenna object
antenna = katpoint.Antenna(first_ant.sensor.observer.get_value())
# Expected beamwidth in radians (beamwidth factor x lambda / D)
expected_width = antenna.beamwidth * katpoint.lightspeed / (opts.centre_freq * 1e6) / antenna.diameter
# Linearly interpolate pointing coordinates to correlator data timestamps
interp = fit.PiecewisePolynomial1DFit(max_degree=1)
interp.fit(az[0], az[1])
az = katpoint.deg2rad(interp(timestamps))
interp.fit(el[0], el[1])
el = katpoint.deg2rad(interp(timestamps))
# Calculate target coordinates (projected az-el coordinates relative to target object)
target_coords = np.vstack(target.sphere_to_plane(az, el, timestamps, antenna))

# Do quick beam + baseline fitting, where both are fitted in 2-D target coord space
# This makes no assumptions about the structure of the scans - they are just viewed as a collection of samples
baseline = fit.Polynomial2DFit((1, 3))
prev_err_power = np.inf
# Initially, all data is considered to be in the "outer" region and therefore forms part of the baseline
outer = np.tile(True, len(power))
print "Fitting quick beam and baseline of degree (1, 3) to target '%s':" % (target.name,)
# Alternate between baseline and beam fitting for a few iterations
for n in xrange(10):
    # Fit baseline to "outer" regions, away from where beam was found
    baseline.fit(target_coords[:, outer], power[outer])
    # Subtract baseline
    bl_resid = power - baseline(target_coords)
    # Fit beam to residual, with the initial beam center at the peak of the residual
    peak_ind = bl_resid.argmax()
    peak_pos = target_coords[:, peak_ind]
    peak_val = bl_resid[peak_ind]
    beam = scape.beam_baseline.BeamPatternFit(peak_pos, expected_width, peak_val)
    beam.fit(target_coords, bl_resid)
    # Calculate Euclidean distance from beam center and identify new "outer" region
    radius = np.sqrt(((target_coords - beam.center[:, np.newaxis]) ** 2).sum(axis=0))
    # This threshold should be close to first nulls of beam - too wide compromises baseline fit
    outer = radius > beam.radius_first_null
    # Check if error remaining after baseline and beam fit has converged, and stop if it has
    resid = bl_resid - beam(target_coords)
    err_power = np.dot(resid, resid)
    print "Iteration %d: residual = %.2f, beam height = %.3f, width = %s, inner region = %d/%d" % \
          (n, (prev_err_power - err_power) / err_power, beam.height, scape.beam_baseline.width_string(beam.width), \
           np.sum(~outer), len(outer))
    if (err_power == 0.0) or (prev_err_power - err_power) / err_power < 1e-5:
        break
    prev_err_power = err_power + 0.0

##### PLOT RESULTS #####

start_time = timestamps.min()
t = timestamps - start_time
plt.plot(t, power, 'b')
plt.plot(t, baseline(target_coords), 'g')
plt.plot(t, beam(target_coords) + baseline(target_coords), 'g')
plt.xlabel('Time in seconds since %s' % katpoint.Timestamp(start_time).local())
plt.ylabel('Power')
plt.title('Quick beam and baseline fit')

print "Beam offset is (%f, %f) deg" % (katpoint.rad2deg(beam.center[0]), katpoint.rad2deg(beam.center[1]))

def set_delay(time_after_now, delay=None):
    t = katpoint.Timestamp() + time_after_now
    if delay is None:
        delay = cable2 - cable1 + tgt.geometric_delay(ant2, t, ant1)[0]
    print delay
    roach.req.poco_delay('0x', int(t.secs) * 1000, '%.9f' % (delay * 1000))
    roach.req.poco_delay('0y', int(t.secs) * 1000, '%.9f' % (delay * 1000))
