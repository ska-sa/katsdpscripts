#!/usr/bin/env python
#
# Track delay calibrator target for a specified time.
# Obtain delay solutions and apply them to the delay tracker in the CBF proxy.
#
# Ludwig Schwardt
# 5 April 2017
#

import time

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


class NoDelaysAvailableError(Exception):
    """No delay solutions are available from the cal pipeline."""


def get_cal_inputs(telstate):
    """Input labels associated with calibration products."""
    if 'cal_antlist' not in telstate or 'cal_pol_ordering' not in telstate:
        return []
    ants = telstate['cal_antlist']
    polprods = telstate['cal_pol_ordering']
    pols = [prod[0] for prod in polprods if prod[0] == prod[1]]
    return [ant + pol for pol in pols for ant in ants]


def get_delaycal_solutions(session):
    """Retrieve delay calibration solutions from telescope state."""
    inputs = get_cal_inputs(session.telstate)
    if not inputs or 'cal_product_K' not in session.telstate:
        return {}
    solutions, solution_ts = session.telstate.get_range('cal_product_K')[0]
    if solution_ts < session.start_time:
        return {}
    return dict(zip(inputs, solutions.real.flat))


# Set up standard script options
usage = "%prog [options] [<'target/catalogue'> ...]"
description = 'Track the source with the highest elevation and calibrate ' \
              'delays based on it. Pick a standard calibrator if none given.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track the source, in seconds (default=%default)')
parser.add_option('--reset', action='store_true', default=False,
                  help='Zero the delay adjustments afterwards')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Delay calibration observation')
# Parse the command line
opts, args = parser.parse_args()

# Default catalogue is a set of primary calibrator targets with flux models
if len(args) == 0:
    args.append('PKS 1934-63 | J1939-6342, radec, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231)')
    args.append('PKS 0408-65 | J0408-6545, radec, 04:08:20.38, -65:45:09.1, (800.0 8400.0 -3.708 3.807 -0.7202)')
    args.append('3C286       | J1331+3030, radec, 13:31:08.29, +30:30:33.0, (800.0 43200.0 0.956 0.584 -0.1644)')

# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        raise NoTargetsUpError("No targets are currently visible - please re-run the script later")
    # Pick source with the highest elevation as our target
    target = observation_sources.sort('el').targets[-1]
    target.add_tags('delaycal single_accumulation')
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        user_logger.info("Target to be observed: %r", target.description)
        session.label('un_corrected')
        session.track(target, duration=opts.track_duration)
        # Attempt to jiggle cal pipeline to drop its delay solutions
        session.ants.req.target('')
        user_logger.info("Waiting for delays to materialise in cal pipeline")
        time.sleep(30)
        sample_rate = 0.0
        delays = {}
        if not kat.dry_run:
            sample_rate = session.telstate.get('cbf_adc_sample_rate', 0.0)
            delays = get_delaycal_solutions(session)
            if not delays:
                raise NoDelaysAvailableError("No delay solutions found in telstate %r"
                                             % (session.telstate,))
        user_logger.info("Delay solutions (sample rate = %f Hz):", sample_rate)
        for inp, delay in delays.items():
            user_logger.info("    %s: %.3f ns (%.3f samples)",
                             inp, delay * 1e9, delay * sample_rate)
        if opts.reset:
            user_logger.info("Zeroing all delay adjustments")
            session.cbf.fengine.req.adjust_all_delays()
        else:
            user_logger.info("Adjusting delays on CBF proxy")
            session.cbf.fengine.req.adjust_all_delays(delays)
