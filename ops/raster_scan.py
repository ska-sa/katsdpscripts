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
parser.add_option('-t', '--tgt', dest='tgt', type="string", default="Takreem,azel,20,30", metavar='TGT',
                  help='Target to scan, as description string (default="%default")')
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

# Setup hardware, do scan with standard settings, and shutdown afterwards
ffobserve.setup(ff, ants)
ffobserve.raster_scan(ff, ants, opts.tgt)
ffobserve.shutdown(ff)

ff.disconnect()
