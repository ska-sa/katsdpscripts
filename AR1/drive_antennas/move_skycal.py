#!/usr/bin/python
#track target(s) for a specified time.

from katcorelib import standard_script_options, verify_and_connect, user_logger
import katpoint
import time
from katpoint import wrap_angle
import numpy as np



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
# Set up standard script options
parser = standard_script_options(usage="%prog [options] hotload or coldload",
                                 description='Perform a mesurement of system tempreture using hot and cold on sky loads'
                                             'Over 6 frequency ranges. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=30.0,
                  help='Length of time for each loading, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float',default=-1,
                  help='Maximum duration of script, in seconds (the default is to observing all sources once)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Hotload and Coldload observation',dump_rate = 1.0/0.512)

# Parse the command line
opts, args = parser.parse_args()
if opts.observer is None :  raise RuntimeError("No observer provided script")

targetlist = []
for argstr in args:
    temp,taz,tel= argstr.split(',')
   
    print "azimuth : %r  Elevation : %r " % (taz, tel)
    targetlist.append([taz,tel])

nd_coupler = {'diode' : 'coupler', 'on' : opts.track_duration, 'off' : 0., 'period' : 0.}

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    # observation_sources = collect_targets(kat, args)
    moon = kat.sources.lookup['moon']
    print "Set Sensor stratergy"
    kat.ants.set_sampling_strategy("lock", "event")
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        if not kat.dry_run :
            user_logger.error("Unable to set Antenna mode to 'STOP'.")


    once = True
    start_time = time.time()
    while once or  time.time() < start_time + opts.max_duration :
        once = False
        moon =  katpoint.Target('Moon, special')
        antenna = katpoint.Antenna('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0, -0:05:30.6 0 -0:00:03.3 0:02:14.2 0:00:01.6 -0:01:30.6 0:08:42.1, 1.22')  # find some way of getting this from session
        moon.antenna = antenna
        off1_azel = katpoint.construct_azel_target(wrap_angle(moon.azel()[0] + np.radians(10) ),moon.azel()[1] )
        off1_azel.antenna = antenna
        off1      = katpoint.construct_radec_target(off1_azel.radec()[0],off1_azel.radec()[1])
        off1.antenna = antenna
        off1.name = 'off1'

        off2_azel = katpoint.construct_azel_target(wrap_angle(moon.azel()[0] - np.radians(10) ),moon.azel()[1] )
        off2_azel.antenna = antenna
        off2      = katpoint.construct_radec_target(off2_azel.radec()[0],off2_azel.radec()[1])
        off2.antenna =  antenna 
        off2.name = 'off2'
        sources = katpoint.Catalogue(add_specials=False)
        sources.add(moon)
        sources.add(off2)
        sources.add(off1)
        txtlist = ', '.join(  [ "'%s'" % (target.name,)  for target in sources])
        user_logger.info("Calibration targets are [%s]" %(txtlist))
        for nd in ['off','on']:
            user_logger.info("Please turn the noise diode %s" %(nd))
            if not kat.dry_run: time.sleep(20)
            for target in sources:
                track(kat.ants,target,duration=0,dry_run=kat.dry_run)
