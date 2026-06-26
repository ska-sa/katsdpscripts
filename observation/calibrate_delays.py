#!/usr/bin/env python
#
# Track delay calibrator target for a specified time.
# Obtain delay solutions and apply them to the delay tracker in the CBF proxy.
#
# Ludwig Schwardt
# 5 April 2017
#

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger, SessionSDP)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track the first source above the horizon and calibrate ' \
              'delays based on it. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=32.0,
                  help='Length of time to track the source for calibration, '
                       'in seconds (default=%default)')
parser.add_option('--verify-duration', type='float', default=64.0,
                  help='Length of time to revisit the source for verification, '
                       'in seconds (default=%default)')
parser.add_option('--fengine-gain', type='int_or_default', default='default',
                  help='Set correlator F-engine gain')
parser.add_option('--fft-shift', type='int_or_default', default='default',
                  help='Set correlator F-engine FFT shift')
parser.add_option('--reset-delays', action='store_true', default=False,
                  help='Zero the delay adjustments afterwards (i.e. check only)')
parser.add_option('--reconfigure-sdp', action="store_true", default=False,
                  help='Reconfigure SDP subsystem at the start to clear '
                       'crashed containers or to load a new version of SDP')
parser.add_option('--timeout', type='float', default=600.0,#default increased from 300s to 600s because 6 minutes was required for c544M4k 1s mode
                  help='Maximum length of time to wait for delays (K-cross diode), '
                       'solutions to be computed by pipeline (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Delay calibration observation that adjusts delays')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('J1939-6342'), description ('radec, 19:39, -63:42') or "
                     "catalogue file name ('three_calib.csv')")

# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    if opts.reconfigure_sdp:
        user_logger.info("Reconfiguring SDP subsystem")
        sdp = SessionSDP(kat)
        sdp.req.product_reconfigure(timeout=300)  # Same timeout as in SDP proxy
    observation_sources = collect_targets(kat, args)
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        # Reset F-engine to a known good state first
        session.set_fengine_fft_shift(opts.fft_shift)
        session.set_fengine_gains(opts.fengine_gain)
        session.adjust_fengine_delays(0)
        # Quit if there are no sources to observe or not enough antennas for cal
        if len(session.ants) < 4:
            raise ValueError('Not enough receptors to do calibration - you '
                             'need 4 and you have %d' % (len(session.ants),))
        sources_above_horizon = observation_sources.filter(el_limit_deg=opts.horizon+5)
        if not sources_above_horizon:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        # Pick the first source that is up (this assumes that the sources in
        # the catalogue are ordered from highest to lowest priority)
        target = sources_above_horizon.targets[0]
        target.add_tags('bfcal single_accumulation')
        session.capture_start()
        session.label('un_corrected')
        user_logger.info("Initiating %g-second track on target %r",
                         opts.track_duration, target.description)
        # Get onto the source
        session.track(target, duration=0, announce=False)
        # Fire noise diode during track
        session.fire_noise_diode(on=opts.track_duration, off=0)
        # Attempt to jiggle cal pipeline to drop its delay solutions
        session.stop_antennas()
        user_logger.info("Waiting for delays to materialise in cal pipeline")
        # Wait for the last relevant bfcal product from the pipeline
        hv_delays = session.get_cal_solutions('KCROSS_DIODE', timeout=opts.timeout)
        delays = session.get_cal_solutions('K')
        # Add HV delay to total delay
        for inp in delays:
            delays[inp] += hv_delays[inp]
        # The main course
        session.adjust_fengine_delays(delays)
        if opts.verify_duration > 0:
            session.label('corrected')
            user_logger.info("Revisiting target %r for %g seconds "
                             "to see if delays are fixed",
                             target.name, opts.verify_duration)
            session.track(target, duration=0, announce=False)
            session.fire_noise_diode(on=opts.verify_duration, off=0)
        if opts.reset_delays:
            session.adjust_fengine_delays(0)
        else:
            # Set last-delay-calibration script sensor on the subarray.
            session.sub.req.set_script_param('script-last-delay-calibration',
                                             kat.sb_id_code)
