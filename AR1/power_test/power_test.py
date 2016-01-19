#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from katcorelib import standard_script_options, verify_and_connect, collect_targets, user_logger
import katpoint
import time


def track(ants,target,index='l',duration=1,dry_run=False):
    # send this target to the antenna.
    ants.req.target(target.description)
    ants.req.mode("POINT")
    for ant_x in ants:
        user_logger.info("Slewing %s to target : %s"%(ant_x.name,target,))
    ants.req.ap_set_indexer_position(index)
    for ant_x in ants:
        user_logger.info("Changing %s indexer to position : %s"%(ant_x.name,index,))
   #if not dry_run : time.sleep(duration)
    
    locks = 0
    for ant_x in ants:
        if ant_x.wait("lock", True, 300): locks += 1
    if len(ants) == locks:
        user_logger.info("Tracking Target : %s for %s seconds"%(target,str(duration)))
        time.sleep(duration)
        user_logger.info("Target tracked : %s "%(target,))
        return True
    else:
        user_logger.warning("Unable to track Targe : %s "%(target,))
        return False


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified in the form "az,el,band" . Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option( '--num-repeat', type='int', default=1,
                  help='The number of times to repeat the sequence (once by by default)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Power Test',dump_rate=0.1)
# Parse the command line
opts, args = parser.parse_args()
if opts.observer is None :  raise RuntimeError("No observer provided script")
if len(args) == 0 : raise RuntimeError("No targets provided to the script")
targetlist = []
for argstr in args:
    taz,tel,band = argstr.split(',')
    targetlist.append([taz,tel,band])

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        if not kat.dry_run : user_logger.error("Unable to set Antenna mode to 'STOP'.")

    for i in range(int(opts.num_repeat)):
        for taz,tel,band in targetlist:
            target = katpoint.Target('Name,azel, %s,%s'%(taz,tel))
            track(kat.ants,target,index=band,duration=opts.track_duration,dry_run=kat.dry_run)
            
#power_test.py "0,15,L"   -t 120
#power_test.py "140,25,L" "160,35,X" --num-repeat=5 -t 1
#power_test.py "140,25,L" -t 1
    
    