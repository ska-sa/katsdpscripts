#!/usr/bin/python
# Perform spiral holography scan on specified target(s). Mostly used for beam pattern measurement.
#
# to run on simulator:
# ssh kat@monctl.comm
# kat-start.sh (may need to call kat-stop.sh, or better kat-kill.py, and possibly kill all screen sessions on kat@monctl.comm, and possibly kat@proxy.monctl.comm)
# ipython
# import katuilib
# configure()
# %run ~/scripts/observation/spiral_holography_scan.py -f 1722 -a ant1,ant2,ant3,ant4,ant5,ant6,ant7 -b ant2,ant3 --num-cycles 1 --cycle-duration 120 -l 12 'AFRISTAR' -o mattieu --sb-id-code='20121030-0003'
# look on http://kat-flap.control.kat.ac.za/kat/KatGUI.swf and connect to 'comm'
#
#using schedule blocks
#help on schedule blocks: https://sites.google.com/a/ska.ac.za/intranet/teams/operators/kat-7-nominal-procedures/frequent-tasks/control-tasks/observation
#to view progress: http://192.168.193.8:8081/tailtask/<sb_id_code>/progress
#to view signal displays remotely safari goto "vnc://kat@right-paw.control.kat.ac.za"
#
#ssh kat@kat-ops.karoo
#ipython
#import katuilib
#configure_obs()
#obs.sb.new_clone('20121203-0013')
#obs.sb.instruction_set="run-obs-script ~/scripts/observation/spiral_holography_scan.py -f 1722 -b ant5 --scan-extent 6 --cycle-duration 6000 --num-cycles 1 --kind 'uniform' '3C 286' --stow-when-done"
#look on http://kat-flap.control.kat.ac.za/kat/KatGUI.swf and connect to 'karoo from site'

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
import katpoint

# Import script helper functions from observe.py
from katcorelib import standard_script_options, verify_and_connect, collect_targets, \
                       start_session, user_logger, ant_array
import numpy as np

#anystowed=np.any([res._returns[0][4]=='STOW' for res in all_ants.req.sensor_value('mode').values()])
def plane_to_sphere_holography(targetaz,targetel,ll,mm):
    scanaz=targetaz-np.arcsin(np.clip(ll/np.cos(targetel),-1.0,1.0))
    scanel=np.arcsin(np.clip((np.sqrt(1.0-ll**2-mm**2)*np.sin(targetel)+np.sqrt(np.cos(targetel)**2-ll**2)*mm)/(1.0-ll**2),-1.0,1.0))
    return scanaz,scanel
    
#same as katpoint.projection._sphere_to_plane_common(az0=scanaz,el0=scanel,az=targetaz,el=targetel) with ll=ortho_x,mm=-ortho_y
def sphere_to_plane_holography(targetaz,targetel,scanaz,scanel):
    #produces direction cosine coordinates from scanning antenna azimuth,elevation coordinates
    #see _coordinate options.py for derivation
    ll=np.cos(targetel)*np.sin(targetaz-scanaz)
    mm=np.cos(targetel)*np.sin(scanel)*np.cos(targetaz-scanaz)-np.cos(scanel)*np.sin(targetel)
    return ll,mm

##repeats start coordinate of pattern by tracktime for stability of scan
#repeats origin by tracktime
#adds extra slewtime at starting point of pattern
def generatebox(totextent,tottime,tracktime=0,slewtime=0,sampletime=1):
    totextent=np.float(totextent)
    tottime=np.float(tottime)
    tracktime=np.float(tracktime)
    slewtime=np.float(slewtime)
    sampletime=np.float(sampletime)
    nperside=np.int((tottime-tracktime-2.0*slewtime)/sampletime/4)
    t=np.linspace(-1,1,nperside,endpoint=False)
    # x=totextent/2.0*np.r_[np.tile(t[0],np.int(tracktime/sampletime)),t,np.ones(nperside),-t,-np.ones(nperside)]
    # y=totextent/2.0*np.r_[np.tile(1.0,np.int(tracktime/sampletime)),np.ones(nperside),-t,-np.ones(nperside),t]
    x=totextent/2.0*np.r_[np.tile(0.0,np.int((tracktime+slewtime)/sampletime)),np.tile(t[0],np.int((slewtime)/sampletime)),t,np.ones(nperside),-t,-np.ones(nperside)]
    y=totextent/2.0*np.r_[np.tile(0.0,np.int((tracktime+slewtime)/sampletime)),np.tile(1.0,np.int((slewtime)/sampletime)),np.ones(nperside),-t,-np.ones(nperside),t]    
    return x,y

##repeats start coordinate of pattern by tracktime for stability of scan
#repeats origin by tracktime
#adds extra slewtime at starting point of pattern
def generateellipse(xextent,yextent,tottime,tracktime=0,slewtime=0,sampletime=1):
    xextent=np.float(xextent)
    yextent=np.float(yextent)
    tottime=np.float(tottime)
    tracktime=np.float(tracktime)
    slewtime=np.float(slewtime)
    sampletime=np.float(sampletime)
    nsamples=np.int((tottime-tracktime-2.0*slewtime)/sampletime)
    
    t=np.linspace(-np.pi,np.pi,nsamples)
    x=xextent/2.0*np.cos(t)
    y=yextent/2.0*np.sin(t)
    # x=np.r_[np.tile(x[0],np.int(tracktime/sampletime)),x]
    # y=np.r_[np.tile(y[0],np.int(tracktime/sampletime)),y]
    x=np.r_[np.tile(0.0,np.int((tracktime+slewtime)/sampletime)),np.tile(x[0],np.int(slewtime/sampletime)),x]
    y=np.r_[np.tile(0.0,np.int((tracktime+slewtime)/sampletime)),np.tile(y[0],np.int(slewtime/sampletime)),y]
    return x,y

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='This script performs a holography scan on the specified target. '
                                             'All the antennas initially track the target, whereafter a subset '
                                             'of the antennas (the "scan antennas" specified by the --scan-ants '
                                             'option) perform a spiral raster scan on the target. Note also some '
                                             '**required** options below.')
# Add experiment-specific options
parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('--off-ants', help='Subset of tracking antennas to apply extra offset to')
parser.add_option('--off-x', default='0', help='List of floats for x direction offsets in arc minutes (default=%default). One x any y offset is used per cycle.')
parser.add_option('--off-y', default='0', help='List of floats for y direction offsets in arc minutes (default=%default). One x any y offset is used per cycle.')
parser.add_option('--num-cycles', type='int', default=1,
                  help='Number of beam measurement cycles to complete (default=%default)')
parser.add_option('--cycle-duration', type='float', default=60.0,
                  help='Time to spend measuring beam pattern per cycle, in seconds (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=1.0,
                  help='Diameter of beam pattern to measure, in degrees (default=%default)')
parser.add_option('--y-extent', type='float', default=None,
                  help='Y extent of scan if kind=ellipse, in degrees (default=%default)')
parser.add_option('--kind', type='string', default='circle',
                  help='Kind of scan, could be "box" or "ellipse" (default=%default)')
parser.add_option('--tracktime', type='float', default=0.0,
                  help='Extra time in seconds for scanning antennas to track when passing over target (default=%default)')
parser.add_option('--slewtime', type='float', default=0.0,
                  help='Extra time in seconds allowed for scanning antennas to slew between target and perimeter (default=%default)')
parser.add_option('--sampletime', type='float', default=1.0,
                  help='time in seconds to spend on pointing (default=%default)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Circular pointing scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if (opts.kind=='box'):
    cx,cy=generatebox(totextent=opts.scan_extent,tottime=opts.cycle_duration,tracktime=opts.tracktime,slewtime=opts.slewtime,sampletime=opts.sampletime)
else:
    if (opts.y_extent!=None):
        cx,cy=generateellipse(xextent=opts.scan_extent,yextent=opts.y_extent,tottime=opts.cycle_duration,tracktime=opts.tracktime,slewtime=opts.slewtime,sampletime=opts.sampletime)
    else:
        cx,cy=generateellipse(xextent=opts.scan_extent,yextent=opts.scan_extent,tottime=opts.cycle_duration,tracktime=opts.tracktime,slewtime=opts.slewtime,sampletime=opts.sampletime)
timeperstep=opts.sampletime;

if len(args) == 0:
    raise ValueError("Please specify a target argument via name ('Ori A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

offsetxlist=[np.float(off) for off in opts.off_x.split(',')]
offsetylist=[np.float(off) for off in opts.off_y.split(',')]
# Check basic command-line options and obtain a kat object connected to the appropriate system
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args)

    # Initialise a capturing session (which typically opens an HDF5 file)
    with start_session(kat, **vars(opts)) as session:
        # Use the command-line options to set up the system
        session.standard_setup(**vars(opts))
        if not opts.no_delays and not kat.dry_run :
            if session.dbe.req.auto_delay('on'):
                user_logger.info("Turning on delay tracking.")
            else:
                user_logger.error('Unable to turn on delay tracking.')
        elif opts.no_delays and not kat.dry_run:
            if session.dbe.req.auto_delay('off'):
                user_logger.info("Turning off delay tracking.")
            else:
                user_logger.error('Unable to turn off delay tracking.')
            if session.dbe.req.zero_delay():
                user_logger.info("Zeroed the delay values.")
            else:
                user_logger.error('Unable to zero delay values.')

        all_ants = session.ants
        # Form scanning antenna subarray (or pick the first antenna as the default scanning antenna)
        scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else session.ants[0], 'scan_ants')
        # Assign rest of antennas to tracking antenna subarray
        track_ants = ant_array(kat, [ant for ant in all_ants if ant not in scan_ants], 'track_ants')
        # Assign offset tracking antennas
        if (opts.off_ants):
            off_ants = ant_array(kat, opts.off_ants, 'track_ants')
        else:
            off_ants = track_ants
        # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
        nd_params = session.nd_params
        session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
        session.capture_start()

        targets_observed = []
        for cycle in range(opts.num_cycles):
            for target in targets.iterfilter(el_limit_deg=opts.horizon+(opts.scan_extent/2.0)):
                session.label('holo')
                user_logger.info("Performing circular scan cycles (%g-second cycles extending %g degrees) on target '%s'"
                                 % (opts.cycle_duration, opts.scan_extent, target.name))
                user_logger.info("Performing scan cycle %d of %d."%(cycle+1, opts.num_cycles))
                session.ants = all_ants
                user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                session.track(target, duration=0, announce=False)#note this actually does nothing useful, doesnt even go to target (keeps existing offset if any exists)
                session.fire_noise_diode(announce=False, **nd_params)#provides opportunity to fire noise diode                
                if (opts.off_ants):#introduce pointing offsets in offset tracking antennas
                    session.ants = off_ants
                    session.ants.req.offset_fixed(offsetxlist[cycle%len(offsetxlist)]/60.0,-offsetylist[cycle%len(offsetylist)]/60.0,opts.projection)
                session.ants = scan_ants
                user_logger.info("Using scan antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
#                session.set_target(target)
#                session.ants.req.drive_strategy('shortest-slew')
#                session.ants.req.mode('POINT')
                if not kat.dry_run:
                    for scan_index in range(len(cx)):
                        lastproctime=time.time()
                        targetaz_rad,targetel_rad=target.azel()
                        scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,cx[scan_index]*np.pi/180.0,cy[scan_index]*np.pi/180.0)
                        # targetx,targety=katpoint.sphere_to_plane[opts.projection](targetaz_rad,targetel_rad,scanaz,scanel)
                        targetx,targety=sphere_to_plane_holography(scanaz,scanel,targetaz_rad,targetel_rad)
                        session.ants.req.offset_fixed(targetx*180.0/np.pi,-targety*180.0/np.pi,opts.projection)
                        # session.ants.req.offset_fixed(cx[scan_index],cy[scan_index],opts.projection)
                        curproctime=time.time()
                        proctime=curproctime-lastproctime
                        if (timeperstep>proctime):
                            time.sleep(timeperstep-proctime)
                        lastproctime=time.time()

                targets_observed.append(target.name)
        user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))

        #set session antennas to all so that stow-when-done option will stow all used antennas and not just the scanning antennas
        session.ants = all_ants
                

