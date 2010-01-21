import ffuilib
import ffobserve
from optparse import OptionParser
import sys

# Parse command-line options that allow the defaults to be overridden
# Default FF configuration is *local*, to prevent inadvertent use of the real hardware
parser = OptionParser(usage="usage: %prog [options]")
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                  help='Telescope configuration file to use in conf directory (default="%default")')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                  help='Selected configuration to use (default="%default")')
parser.add_option('-a', '--ants', dest='ants', type="string", metavar='ANTS',
                  help="Comma-separated list of antennas to include in scan (e.g. 'ant1,ant2')," +
                       " or 'all' for all antennas - this MUST be specified (safety reasons)")
parser.add_option('-f', '--centre_freq', dest='centre_freq', type="float", default=1822.0, metavar='AZ',
                  help='Centre frequency, in MHz (default="%default")')
parser.add_option('-z', '--az', dest='az', type="float", default=168.0, metavar='AZ',
                  help='Azimuth angle along which to do tipping curve, in degrees (default="%default")')
(opts, args) = parser.parse_args()

# Force antennas to be specified to sensitise the user to what will physically move
if opts.ants is None:
    print 'Please specify the antennas to use via -a option (yes, this is a non-optional option...)'
    sys.exit(1)

# Build Fringe Finder configuration, as specified in user-facing config file
# This connects to all the proxies and devices and queries their commands and sensors
ff = ffuilib.tbuild(opts.ini_file, opts.selected_config)

# Create a list of the specified antenna devices, and complain if they are not found
if opts.ants.strip() == 'all':
    ants = ff.ants.devs
else:
    try:
        ants = [getattr(ff, ant_x.strip()) for ant_x in opts.ants.split(",")]
    except AttributeError:
        raise ValueError("Antenna '%s' not found" % ant_x)

ffobserve.setup(ff, ants, opts.centre_freq)
# Iterate through elevation angles
for el in [2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
    ffobserve.track(ff, ants, "azel, %f, %f" % (opts.az, el), duration=15.0)
    ffobserve.fire_noise_diode(ff, ants, 'coupler', scan_id=2)
ffobserve.shutdown(ff)

ff.disconnect()
