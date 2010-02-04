import katpoint
import ffuilib
import ffobserve
from optparse import OptionParser
import sys

# Parse command-line options that allow the defaults to be overridden
# Default FF configuration is *local*, to prevent inadvertent use of the real hardware
parser = OptionParser(usage="usage: %prog [options]")
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", metavar='INI',
                  help='Telescope configuration file to use in conf directory (default="%default")')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string",  metavar='SELECTED',
                  help='Selected configuration to use (default="%default")')
parser.add_option('-p', '--printonly', dest='printonly', action="store_true", default=False,
                  help="Do not actually observe, only print, if switch included (default=%default)")

(opts, args) = parser.parse_args()

# Force antennas to be specified to sensitise the user to what will physically move
if opts.ini_file is None or opts.selected_config is None:
    print 'Please specify the configuration file and selection using -i and -s'
    sys.exit(0)

# Build Fringe Finder configuration, as specified in user-facing config file
# This connects to all the proxies and devices and queries their commands and sensors
ff = ffuilib.tbuild(opts.ini_file, opts.selected_config)

# Load catalogue of point sources
cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
# Prune the catalogue to only contain sources that are good for baseline calibration
great_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A', 'Pictoris A']
good_sources =  ['3C48', '3C84', 'J0408-6545', 'J0522-3627', '3C161', 'J1819-6345', 'J1939-6342', '3C433', 'J2253+1608']
good_cat = katpoint.Catalogue([cat[src] for src in great_sources + good_sources], add_specials=False, antenna=cat.antenna)

compscan_id = 0
start_time = katpoint.Timestamp()

if opts.printonly:
    # Only list targets that will be visited
    for target in good_cat.iterfilter(el_limit_deg=5, timestamp=start_time):
        print "Compound scan", compscan_id, "will observe", target.name
        compscan_id += 1
        start_time += 120.0
else:
    try:
        # The real experiment
        ffobserve.setup(ff, ff.ants, centre_freq=1800.0)

        # Observe at least 1.5 hours
        while katpoint.Timestamp() - start_time < 4500.0:
            for target in good_cat.iterfilter(el_limit_deg=5):
                ffobserve.track(ff, ff.ants, target, duration=120.0, compscan_id=compscan_id, drive_strategy='longest-track')
                compscan_id += 1

    finally:
        # ALWAYS run shutdown, even on an error or Keyboard Interrupt, as your hard drive will fill up otherwise...
        ffobserve.shutdown(ff)
        ff.disconnect()
