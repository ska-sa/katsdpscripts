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

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=ff.sources.antenna)
good_sources = ['Cygnus A', '3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A', 'Hercules A', 'Sagittarius A', '3C353', 'J1819-6345', '3C279', '3C286','3C295','J2253+1608','Pictoris A','Fornax A','J0522-3627','Cassiopeia A','3C161','3C48','3C147']
good_cat = katpoint.Catalogue([cat[src] for src in good_sources], add_specials=False, antenna=cat.antenna)

if not opts.printonly:
    ffobserve.setup(ff, ff.ants, centre_freq=1800.0)

compscan_id = 0
for target in good_cat.iterfilter(el_limit_deg=5):
    if opts.printonly:
        print "At compscan_id", compscan_id, "will observe",target
    else:
        ffobserve.track(ff, ff.ants, target.description, duration=120.0, compscan_id=compscan_id, drive_strategy='longest-track')
    compscan_id += 1

if not opts.printonly:
    ffobserve.shutdown(ff)

ff.disconnect()

