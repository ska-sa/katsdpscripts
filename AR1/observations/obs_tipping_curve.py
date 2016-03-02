#!/usr/bin/python
# Perform tipping curve scan  and find a specified azimith if one is not give.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement
import time
from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger
import numpy as np
import katpoint
# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform tipping curve scan for a specified azimuth position. \
                                              Or Select a Satilite clear Azimith,\
                                              Some options are **required**.")
# Add experiment-specific options
parser.add_option('-z', '--az', type="float", default=None,
                  help='Azimuth angle along which to do tipping curve, in degrees (default="%default")')
parser.add_option('--spacing', type="float", default=1.0,
                  help='The Spacing along the elevation axis of the tipping curve that measuremnts are taken, in degrees (default="%default")')
parser.add_option( '--tip-both-directions', action="store_true" , default=False,
                  help='Do tipping curve from low to high elevation and then from high to low elevation')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays (default="%default")')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Tipping Curve')

# Parse the command line
opts, args = parser.parse_args()

on_time = 15.0
with verify_and_connect(opts) as kat:
    # Start Antenna for observing
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(3)
    else:
        user_logger.error("Unable to set Antenna mode to 'STOP'.")
    # Ensure that azimuth is in valid physical range of -185 to 275 degrees
    if opts.az is None:
        user_logger.info("No Azimuth selected , selecting clear Azimith")
        if not kat.dry_run:
            timestamp = [katpoint.Timestamp(time.time()) for i in range(int((np.arange(15.0,90.1,opts.spacing).shape[0]*(on_time+20.0+1.0))))]
            #load the standard KAT sources ... similar to the SkyPlot of the katgui
            observation_sources = kat.sources
            source_az = []
            for source in observation_sources.targets:
                az, el = np.degrees(source.azel(timestamp=timestamp))   # was rad2deg
                az[az > 180] = az[az > 180] - 360
                source_az += list(set(az[el > 15]))
            source_az.sort()
            gap = np.diff(source_az).argmax()+1
            opts.az = (source_az[gap] + source_az[gap+1]) /2.0
            user_logger.info("Selecting Satillite clear Azimuth=%f"%(opts.az,))
        else:
            opts.az = 0
            user_logger.info("Selecting dummey Satillite clear Azimuth=%f"%(opts.az,))
    else:
        if (opts.az < -185.) or (opts.az > 275.):
            opts.az = (opts.az + 180.) % 360. - 180.
        user_logger.info("Tipping Curve at Azimuth=%f"%(opts.az,))
    user_logger.info("Tipping Curve at Azimuth=%f"%(opts.az,))

    with start_session(kat, **vars(opts)) as session:
        if not opts.no_delays and not kat.dry_run :
            if session.data.req.auto_delay('on'):
                user_logger.info("Turning on delay tracking.")
            else:
                user_logger.error('Unable to turn on delay tracking.')
        elif opts.no_delays and not kat.dry_run:
            if session.data.req.auto_delay('off'):
                user_logger.info("Turning off delay tracking.")
            else:
                user_logger.error('Unable to turn off delay tracking.')
            if session.data.req.zero_delay():
                user_logger.info("Zeroed the delay values.")
            else:
                user_logger.error('Unable to zero delay values.')
        session.standard_setup(**vars(opts))
        session.capture_start()
        # Iterate through elevation angles
        # spacings = list(np.arange(15.0,90.1,opts.spacing))
# RvR -- temporary measure for M063 until AP can handle not being able to reach # target gracefully
        spacings = list(np.arange(15.0,89.1,opts.spacing))
        if opts.tip_both_directions :
            # spacings += list(np.arange(90.0,19.9,-opts.spacing))
            spacings += list(np.arange(89.0,19.9,-opts.spacing))
# RvR -- temporary measure for M063 until AP can handle not being able to reach # target gracefully
        for el in spacings:
            session.label('track')
            session.track('azel, %f, %f' % (opts.az, el), duration=on_time)
            session.fire_noise_diode('coupler', on=10, off=10)
