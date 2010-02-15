import katpoint
import ffuilib
from ffuilib import CaptureSession

import uuid
from optparse import OptionParser
import sys

# Parse command-line options that allow the defaults to be overridden
# Default FF configuration is *local*, to prevent inadvertent use of the real hardware
parser = OptionParser(usage="usage: %prog [options]")
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                  help='Telescope configuration file to use in conf directory (default="%default")')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                  help='Selected configuration to use (default="%default")')
parser.add_option('-u', '--experiment_id', dest='experiment_id', type="string",
                  help='Experiment ID used to link various parts of experiment together (randomly generated UUID by default)')
parser.add_option('-o', '--observer', dest='observer', type="string", help='Name of person doing the observation')
parser.add_option('-d', '--description', dest='description', type="string", default="Baseline calibration",
                  help='Description of observation (default="%default")')
parser.add_option('-a', '--ants', dest='ants', type="string", metavar='ANTS',
                  help="Comma-separated list of antennas to include in scan (e.g. 'ant1,ant2')," +
                       " or 'all' for all antennas - this MUST be specified (safety reasons)")
parser.add_option('-p', '--printonly', dest='printonly', action="store_true", default=False,
                  help="Do not actually observe, only print, if switch included (default=%default)")

(opts, args) = parser.parse_args()

if opts.ants is None:
    print 'Please specify the antennas to use via -a option (yes, this is a non-optional option...)'
    sys.exit(1)
if opts.observer is None:
    print 'Please specify the observer name via -o option (yes, this is a non-optional option...)'
    sys.exit(1)
if opts.experiment_id is None:
    # Generate unique random string via RFC 4122 version 4
    opts.experiment_id = str(uuid.uuid4())

ff = ffuilib.tbuild(opts.ini_file, opts.selected_config)

# Prune the standard catalogue to only contain sources that are good for baseline calibration
great_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273', 'Virgo A', 'Centaurus A', 'Pictoris A']
good_sources =  ['3C48', '3C84', 'J0408-6545', 'J0522-3627', '3C161', 'J1819-6345', 'J1939-6342', '3C433', 'J2253+1608']
good_cat = katpoint.Catalogue([ff.sources[src] for src in great_sources + good_sources],
                              add_specials=False, antenna=ff.sources.antenna)

start_time = katpoint.Timestamp()

if opts.printonly:
    for compscan, target in enumerate(good_cat.iterfilter(el_limit_deg=5, timestamp=start_time)):
        print "Compound scan", compscan, "will observe", target.name
        start_time += 120.0

else:
    # The real experiment
    with CaptureSession(ff, opts.experiment_id, opts.observer, opts.description, opts.ants, centre_freq=1800.0) as session:
        # Observe at least 1.5 hours
        while katpoint.Timestamp() - start_time < 4500.0:
            for target in good_cat.iterfilter(el_limit_deg=5):
                session.track(target, duration=120.0, drive_strategy='longest-track')
