#!/usr/bin/python
# Track SCP  for a specified time for hotbox tests.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect,  start_session, user_logger
import katpoint
from katpoint import wrap_angle
import numpy as np

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

if  not opts.description == 'Hotload and Coldload observation' :  opts.description = 'Hotload and Coldload observation:'+ opts.description

nd_off     = {'diode' : 'coupler', 'on' : 0., 'off' : 0., 'period' : -1.}
nd_coupler = {'diode' : 'coupler', 'on' : opts.track_duration, 'off' : 0., 'period' : 0.}

#if len(args) == 0:
#    raise ValueError("Please specify the sources to observe as arguments, either as "
#                     "description strings or catalogue filenames")

with verify_and_connect(opts) as kat:
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        user_logger.error("Unable to set Antenna mode to 'STOP'.")
    moon = kat.sources.lookup['moon']
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.nd_params = nd_off
        session.capture_start()
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
            for target in sources:
                session.nd_params = nd_off
                for nd in [nd_coupler]:
                    session.nd_params = nd_off
                    session.track(target, duration=0) # get onto the source
                    user_logger.info("Now capturing data - diode %s on" % nd['diode'])
                    session.label('%s'%(nd['diode']))
                    if not session.fire_noise_diode(announce=True, **nd) : user_logger.error("Noise Diode did not Fire , (%s did not fire)" % nd['diode']  )
                session.nd_params = nd_off
                user_logger.info("Now capturing data - noise diode off")
                session.label('track')
                session.track(target, duration=opts.track_duration)
        if opts.max_duration and time.time() > start_time + opts.max_duration:
            user_logger.info('Maximum script duration (%d s) exceeded, stopping script' % (opts.max_duration,))
