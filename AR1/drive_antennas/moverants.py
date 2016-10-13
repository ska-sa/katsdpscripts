#!/usr/bin/python
#track target(s) for a specified time.

from katcorelib import standard_script_options, verify_and_connect, collect_targets, user_logger
import katpoint
import time


def track(ants, target, duration=10, dry_run=False):
    print "Target  :",target
    # send this target to the antenna.
    ants.req.target(target.description)
    print "Target desc. : ",target.description
    ant_names = ','.join([ant.name for ant in ants])
    ants.req.mode("POINT")
    user_logger.info("Slewing %s to target : %r" % (ant_names, target))
    #Wait for antenna to lock onto target
    locks=0
    if not dry_run :
    	for ant_x in ants:
            if ant_x.wait('lock', True, timeout=300): locks +=1
            print "locks status:", ant_x.name , locks,len(ants)
    else:
        locks = len(ants)
    if len(ants)==locks:
        user_logger.info("Target reached : %r wait for %d seconds before slewing to the next target" % (target, duration))
        if not dry_run : time.sleep(duration)
        user_logger.info("Test complete : %r" % (target,))
        return True
    else:
        user_logger.warning("Unable to track Target : %r Check %s sensors  " % (target,ant_names))
        return False


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified in the form "az,el" . Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=30.0,
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
    temp,taz,tel= argstr.split(',')
   
    print "azimuth : %r  Elevation : %r " % (taz, tel)
    targetlist.append([taz,tel])

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
   # observation_sources = collect_targets(kat, args)
    print "Set Sensor stratergy"
    kat.ants.set_sampling_strategy("lock", "event")
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        if not kat.dry_run :
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

    for i in range(int(opts.num_repeat)):
        for taz,tel in targetlist:
            target = katpoint.Target('Name,azel, %s,%s'%(taz,tel))
            track(kat.ants,target,duration=30,dry_run=kat.dry_run)
            print target

