#!/usr/bin/python
#track target(s) for a specified time.

from katcorelib import standard_script_options, verify_and_connect,  user_logger , ant_array
import katpoint
import time

import numpy as np
#import scipy
#from scikits.fitting import NonLinearLeastSquaresFit

#anystowed=np.any([res._returns[0][4]=='STOW' for res in all_ants.req.sensor_value('mode').values()])


def set_target (ants, target):
    ants.req.target(target.description)
    

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


def scan(ants, target, duration=30.0, start=(-3.0, 0.0), end=(3.0, 0.0),
             index=-1, dry_run=False,projection = ('ARC', 0., 0.)):
    # Convert description string to target object, or keep object as is

    if not isinstance(target, katpoint.Target):
      target = katpoint.Target(target)

    scan_name = 'scan' if index < 0 else 'scan %d' % (index,)

    
    user_logger.info("Initiating %g-second scan across target '%s'" %
                       (duration, target.name))
    # This already sets antennas in motion if in mode POINT
    set_target(ants, target)
    
    user_logger.info('slewing to start of %s', scan_name)
    # Move each antenna to the start position of the scan
    ants.req.scan_asym(start[0], start[1], end[0], end[1],duration, projection)
    ants.req.mode('POINT')
    # Wait until they are all in position (with 5 minute timeout)
    locks = 0
    if not dry_run :
    	for ant_x in ants:
            if ant_x.wait('lock', True, timeout=300): locks +=1
            print "locks status:", ant_x.name , locks,len(ants)
    else:
        locks = len(ants)
    if len(ants)==locks:
        user_logger.info('start of %s reached' % (scan_name,))
        if not dry_run : 
            #time.sleep(duration)
            user_logger.info('performing %s' % (scan_name,))
            # Start scanning the antennas
            ants.req.mode('SCAN')
            # Wait until they are all finished scanning (with 5 minute timeout)
            user_logger.info('%s complete' % (scan_name,))
            scanc = 0
            for ant_x in ants:
                if ant_x.wait('scan_status', 'after', timeout=300):
                    scanc +=1
                print "scan status:", ant_x.name , scanc,len(ants) 
        return True
    else:
        user_logger.warning("Unable to Scan Target : %r Check %s sensors  " % (target,','.join([ant.name for ant in ants])))
        return False
 
    return True

 
 
 
def raster_scan(ants,target, num_scans=3, scan_duration=30.0,
                    scan_extent=6.0, scan_spacing=0.5, scan_in_azimuth=True,
                    projection=('ARC', 0., 0.), dry_run=False):
    # Create references to allow easy copy-and-pasting from this function
    # Convert description string to target object, or keep object as is
    if not isinstance(target, katpoint.Target):
     target = katpoint.Target(target)

    
    user_logger.info("Initiating raster scan (%d %g-second scans extending %g degrees) on target '%s'" %
                      (num_scans, scan_duration, scan_extent, target.name))
    # Create start and end positions of each scan, based on scan parameters
    scan_levels = np.arange(-(num_scans // 2), num_scans // 2 + 1)
    scanning_coord = (scan_extent / 2.0) * (-1) ** scan_levels
    stepping_coord = scan_spacing * scan_levels
    # Flip sign of elevation offsets to ensure that the first scan always
    # starts at the top left of target
    scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)
    scan_ends = zip(-scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, scanning_coord)

    # Perform multiple scans across the target
    for scan_index, (start, end) in enumerate(zip(scan_starts, scan_ends)):
        scan(ants,target, duration=scan_duration, start=start, end=end,
                  index=scan_index, projection=projection,)
    return True

 


# Set up standard script options
parser = standard_script_options(usage="%prog [options] 'target'",
                                 description='This script performs a holography scan on the specified target. '
                                             'All the antennas initially track the target, whereafter a subset '
                                             'of the antennas (the "scan antennas" specified by the --scan-ants '
                                             'option) perform a spiral raster scan on the target. Note also some '
                                             '**required** options below.')
parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('--num-cycles', type='int', default=1,
                  help='Number of beam measurement cycles to complete (default=%default)')
parser.add_option('--cycle-duration', type='float', default=300.0,
                  help='Time to spend measuring beam pattern per cycle, in seconds (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=4.0,
                  help='Diameter of beam pattern to measure, in degrees (default=%default)')
parser.add_option('--kind', type='string', default='uniform',
                  help='Kind of spiral, could be "uniform" or "dense-core" (default=%default)')
parser.add_option('--tracktime', type='float', default=1.0,
                  help='Extra time in seconds for scanning antennas to track when passing over target (default=%default)')
parser.add_option('--sampletime', type='float', default=1.0,
                  help='time in seconds to spend on pointing (default=%default)')
parser.add_option('--mirrorx', action="store_true", default=False,
                  help='Mirrors x coordinates of pattern (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Spiral holography scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()
if opts.observer is None :  raise RuntimeError("No observer provided script")

if len(args) == 0:
    raise ValueError("Please specify a target argument via name ('Ori A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")
else:
    target = katpoint.Target(args[0])


# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
   # observation_sources = collect_targets(kat, args)
    print "Set Sensor stratergy"
    lasttargetel=target.azel()[1]*180.0/np.pi
    kat.ants.set_sampling_strategy("lock", "event")
    kat.ants.set_sampling_strategy("scan_status", "event")
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        if not kat.dry_run :
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

    all_ants = kat.ants
    # Form scanning antenna subarray (or pick the first antenna as the default scanning antenna)
    scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else kat.ants[0], 'scan_ants')
    # Assign rest of antennas to tracking antenna subarray
    track_ants = ant_array(kat, [ant for ant in all_ants if ant not in scan_ants], 'track_ants')
    # Disable noise diode by default (to prevent it firing on scan antennas only during scans)

    user_logger.info("Initiating spiral holography scan cycles (%d %g-second cycles extending %g degrees) on target '%s'"
                     % (opts.num_cycles, opts.cycle_duration, opts.scan_extent, target.name))
    raster_scan(scan_ants,target, num_scans=13, scan_duration=120.0,
                        scan_extent=3.0, scan_spacing=0.1,
                        projection=opts.projection, dry_run=kat.dry_run)


#For the raster scans I suggest delta_elev<=0.1 deg 
#and have at least 120sec per az scan over 3deg. 
#With e.g. 13 scans separated by 0.1 deg we will span +- 0.6 deg in elevation.
# This would be 26 min per raster, times 2 for both antennas times 2 for both bands -> 2 hours.
#If more or slower scans can be done, even better. (I know that this is already asking for much!)
