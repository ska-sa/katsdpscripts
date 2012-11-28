#!/usr/bin/python
# Perform radial holography scan on specified target(s). Mostly used for beam pattern mapping.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
#see session2.py track() in katcorelib

# Import script helper functions from observe.py
from katcorelib import standard_script_options, verify_and_connect, collect_targets, \
                       start_session, user_logger, ant_array
import numpy as np
import scipy
from scikits.fitting import NonLinearLeastSquaresFit, PiecewisePolynomial1DFit

def spiral(params,indep):
    x0=indep[0]
    y0=indep[1]
    r=params[0]
    x=r*np.cos(2.0*np.pi*r)
    y=r*np.sin(2.0*np.pi*r)
    return np.sqrt((x-x0)**2+(y-y0)**2)

#note that we want spiral to only extend to above horizon for first few scans in case source is rising
#should test if source is rising or setting before each composite scan, and use -compositey if setting
#singlepointatorigin is True when calculating sampling coordinates, but False when observing
#typically ntime should =narm
#note radextent is half the full extent, and ntime is half the full 'scan time'
#narms is number of radial arms of radextent, performed in ntime
def generatespiral(radextent,ntime,narms,kind='other',singlepointatorigin=False):
    if (kind=='simple'):
        armrad=radextent*(linspace(0,1,ntime))
        armtheta=linspace(0,np.pi,ntime)
        armx=armrad*np.cos(armtheta)
        army=armrad*np.sin(armtheta)
    elif (kind=='approx'):
        armrad=radextent*(linspace(0,1,ntime))
        armtheta=linspace(0,np.pi,ntime)
        armx=armrad*np.cos(armtheta)
        army=armrad*np.sin(armtheta)
        dist=sqrt((armx[:-1]-armx[1:])**2+(army[:-1]-army[1:])**2)
        narmrad=np.cumsum(np.concatenate([np.array([0]),1.0/dist]))
        narmrad*=radextent/max(narmrad)
        narmtheta=narmrad/radextent*np.pi
        armx=narmrad*np.cos(narmtheta)
        army=narmrad*np.sin(narmtheta)
    else:
        armx=np.zeros(ntime)
        army=np.zeros(ntime)
        #must be on curve x=t*cos(np.pi*t),y=t*sin(np.pi*t)
        #intersect (x-x0)**2+(y-y0)**2=1/ntime**2 with spiral
        lastr=0.0
        for it in range(1,ntime):
            data=np.array([1.0/ntime])
            indep=np.array([armx[it-1],army[it-1]])#last calculated coordinate in arm, is x0,y0
            initialparams=np.array([lastr+1.0/ntime]);
            fitter=NonLinearLeastSquaresFit(spiral,initialparams)
            fitter.fit(indep,data)
            lastr=fitter.params[0];
            armx[it]=lastr*np.cos(2.0*np.pi*lastr)
            army[it]=lastr*np.sin(2.0*np.pi*lastr)

        maxrad=np.sqrt(armx[it]**2+army[it]**2)
        armx=armx*radextent/maxrad
        army=army*radextent/maxrad
    # ndist=sqrt((armx[:-1]-armx[1:])**2+(army[:-1]-army[1:])**2)
    # print ndist

    if (singlepointatorigin):
        compositex=np.array([0.0])
        compositey=np.array([0.0])
    else:
        compositex=np.array([])
        compositey=np.array([])
    reverse=False
    for ia in range(narms):
        rot=-ia*np.pi*2.0/narms
        x=armx*np.cos(rot)-army*np.sin(rot)
        y=armx*np.sin(rot)+army*np.cos(rot)
        if reverse:
            reverse=False
            x=x[:0:-1]#omit the final 0
            y=y[:0:-1]#omit the final 0
        else:
            reverse=True
            if (singlepointatorigin):
                x=x[1:]
                y=y[1:]
        compositex=np.concatenate([compositex,x])
        compositey=np.concatenate([compositey,y])
    return compositex,compositey


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='This script performs a holography scan on one or more targets. '
                                             'All the antennas initially track the target, whereafter a subset '
                                             'of the antennas (the "scan antennas" specified by the --scan-ants '
                                             'option) perform a spiral raster scan on the target. Note also some '
                                             '**required** options below.')
# Add experiment-specific options
parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('-k', '--num-scans', type='int', default=10,
                  help='Number of scans across target (default=%default)')
parser.add_option('-t', '--scan-duration', type='float', default=30.0,
                  help='Minimum duration of each scan across target, in seconds (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=6.0,
                  help='Length of each scan, in degrees (default=%default)')
parser.add_option('--num-cycles', type='int', default=1,
                  help='Number of beam map cycles to complete (default=%default)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Spiral holography scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

xx,yy=generatespiral(radextent=opts.scan_extent/2.0,ntime=int(opts.scan_duration/2),narms=int(opts.num_scans*2),kind='other',singlepointatorigin=False)
timeperstep=1.0;

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Ori A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

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
        # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
        nd_params = session.nd_params
        session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
        session.capture_start()

        targets_observed = []
        for cycle in range(opts.num_cycles):
            for target in targets.iterfilter(el_limit_deg=opts.horizon+(opts.scan_extent/2.0)):
                # The entire sequence of commands on the same target forms a single compound scan
                session.label('holo')
                user_logger.info("Initiating holography scan (%d %g-second scans extending %g degrees) on target '%s'"
                                 % (opts.num_scans, opts.scan_duration, opts.scan_extent, target.name))
                user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
                # Slew all antennas onto the target (don't spend any more time on it though)
                session.ants = all_ants
                session.track(target, duration=0, announce=False)
                # Provide opportunity for noise diode to fire on all antennas
                session.fire_noise_diode(announce=False, **nd_params)
                # Perform multiple scans across the target at various angles with the scan antennas only
                session.ants = scan_ants
                user_logger.info("Using scan antennas: %s" % (' '.join([ant.name for ant in session.ants]),))
#                session.set_target(target)
#                session.ants.req.drive_strategy('shortest-slew')
#                session.ants.req.mode('POINT')
                for scan_index in range(len(xx)):
                    session.ants.req.offset_fixed(xx[scan_index],yy[scan_index],opts.projection);
                    time.sleep(timeperstep)

                targets_observed.append(target.name)
        user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
