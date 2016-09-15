#!/usr/bin/env python
#
# Observe either 1934-638 or 0408-65 to establish some basic health 
# Properties of the MeerKAT AR1 system.

import time

import numpy as np

import katpoint

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)
from katsdptelstate import TelescopeState


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


class NoGainsAvailableError(Exception):
    """No gain solutions are available from the cal pipeline."""


def get_cbf_inputs(data):
    """Input labels associated with correlator."""
    reply = data.req.cbf_input_labels()
    return reply.messages[0].arguments[1:] if reply.succeeded else []


def get_telstate(data, sub):
    """Get TelescopeState object associated with current data product."""
    subarray_product = 'array_%s_%s' % (sub.sensor.sub_nr.get_value(),
                                        sub.sensor.product.get_value())
    reply = data.req.spmc_telstate_endpoint(subarray_product)
    return TelescopeState(reply.messages[0].arguments[1]) if reply.succeeded else {}


def get_delaycal_solutions(telstate):
    """Retrieve delay calibration solutions from telescope state."""
    if 'cal_antlist' not in telstate or 'cal_product_K' not in telstate:
        return {}
    ants = telstate['cal_antlist']
    inputs = [ant + pol for pol in 'hv' for ant in ants]
    solutions = telstate['cal_product_K']
    return dict(zip(inputs, solutions.real.flat))


def get_bpcal_solutions(telstate):
    """Retrieve bandpass calibration solutions from telescope state."""
    if 'cal_antlist' not in telstate or 'cal_product_B' not in telstate:
        return {}
    ants = telstate['cal_antlist']
    inputs = [ant + pol for pol in 'hv' for ant in ants]
    solutions = telstate['cal_product_B']
    return dict(zip(inputs, solutions.reshape((solutions.shape[0], -1)).T))


def get_gaincal_solutions(telstate):
    """Retrieve gain calibration solutions from telescope state."""
    if 'cal_antlist' not in telstate or 'cal_product_G' not in telstate:
        return {}
    ants = telstate['cal_antlist']
    inputs = [ant + pol for pol in 'hv' for ant in ants]
    solutions = telstate['cal_product_G']
    return dict(zip(inputs, solutions.flat))

# temporary hack to ensure antenna does not timeout for the moment
def bad_ar1_alt_hack(target, duration, limit=88.):
    [az, el] = target.azel()
    delta_transit = duration * (15. / 3600.)
    if (numpy.rad2deg(float(el)) + delta_transit + delta_transit) > limit:
        return True
    return False

# Set up standard script options
usage = "%prog"
description = 'Observe either 1934-638 or 0408-65 to establish some basic health ' \
              'Properties of the MeerKAT AR1 system.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('--default-gain', type='int', default=200,
                  help='Default correlator F-engine gain (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='basic_health', nd_params='off', project_id='MKAIV-308',
                    description='Basic health test of the system.',horizon=25,track_duration=30)
# Parse the command line
opts, args = parser.parse_args()

# Very bad hack to circumvent SB verification issues
# with anything other than session objects (e.g. kat.data).
# The *near future* will be modelled CBF sessions.
# The *distant future* will be fully simulated sessions via kattelmod.
#if opts.dry_run:
#    import sys
#    sys.exit(0)

# set of targets with flux models    
J1934 = 'PKS 1934-63 | J1939-6342, radec bfcal single_accumulation, 19:39:25.03, -63:42:45.7, (200.0 12000.0 -11.11 7.777 -1.231)'
J0408 = 'PKS 0408-65 | J0408-6545, radec bfcal single_accumulation, 4:08:20.38, -65:45:09.1, (800.0 8400.0 -3.708 3.807 -0.7202)'

# ND states
nd_off = {'diode': 'coupler', 'on': 0., 'off': 0., 'period': -1.}
nd_on = {'diode': 'coupler', 'on': opts.track_duration, 'off': 0., 'period': 0.}


# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    observation_sources.add(J1934)
    observation_sources.add(J0408)
    # Quit early if there are no sources to observe ... use 25 degrees to alow time to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        raise NoTargetsUpError("No targets are currently visible - please re-run the script later")
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        inputs = get_cbf_inputs(session.data)
        # Assume correlator stream is bc product name without the 'b'
        product = kat.sub.sensor.product.get_value()
        corr_stream = product if product.startswith('c') else product[1:]
        session.data.req.capture_start(corr_stream)

        for target in observation_sources.sort('el')[0]:
            # Calibration tests
            user_logger.info("Performing calibration tests")
            if target.flux_model is None:
                user_logger.warning("Target has no flux model (katsdpcal will need it in future)")
            user_logger.info("Resetting F-engine gains to %g to allow phasing up"
                             % (opts.default_gain,))
            for inp in inputs:
                session.data.req.cbf_gain(inp, opts.default_gain)
            session.label('calibration')
            user_logger.info("Initiating %g-second track on target '%s'" %
                             (opts.track_duration, target.name,))
            session.track(target, duration=opts.track_duration, announce=False)
            # Attempt to jiggle cal pipeline to drop its gains
            session.ants.req.target('')
            user_logger.info("Waiting for gains to materialise in cal pipeline")
            session.track('', duration=180, announce=False)
            #time.sleep(180)
            telstate = get_telstate(session.data, kat.sub)
            delays = get_delaycal_solutions(telstate)
            bp_gains = get_bpcal_solutions(telstate)
            gains = get_gaincal_solutions(telstate)
            if not gains:
                raise NoGainsAvailableError("No gain solutions found in telstate %r" % (telstate,))
            user_logger.info("Setting F-engine gains to phase up antennas")
            for inp in set(inputs) and set(gains):
                orig_weights = gains[inp]
                if inp in bp_gains:
                    orig_weights *= bp_gains[inp]
                if inp in delays:
                    # XXX Hacky hack
                    centre_freq = 1284e6
                    num_chans = 32768 if product.endswith('32k') else 4096
                    sideband = 1
                    channel_width = 856e6 / num_chans
                    channel_freqs = centre_freq + sideband * channel_width * (np.arange(num_chans) - num_chans / 2)
                    delay_weights = np.exp(2.0j * np.pi * delays[inp] * channel_freqs)
                    # Guess which direction to apply delays as katcal has a bug here
                    orig_weights *= delay_weights
                amp_weights = np.abs(orig_weights)
                phase_weights = orig_weights / amp_weights
                # Cop out on the gain amplitude but at least correct the phase
                new_weights = opts.default_gain * phase_weights.conj()
                weights_str = [('%+5.3f%+5.3fj' % (w.real, w.imag)) for w in new_weights]
                session.data.req.cbf_gain(inp, *weights_str)
            user_logger.info("Revisiting target %r for %g seconds to see if phasing worked" %
                             (target.name, opts.track_duration))
            session.track(target, duration=opts.track_duration, announce=False)

            # interferometric pointing
            user_logger.info("Performing interferometric pointing tests")
            session.label('interferometric_pointing')
            session.track(target, duration=opts.track_duration, announce=False)
            for direction in {'x', 'y'}:
                for offset in np.linspace(-opts.max_extent, opts.max_extent, opts.number_of_steps // 2):
                    if direction == 'x':
                        offset_target = [offset, 0.0]
                    else:
                        offset_target = [0.0, offset]
                    user_logger.info("Initiating %g-second track on target '%s'" %
                                     (opts.track_duration, target.name,))
                    user_logger.info("Offset of %f,%f degrees " % (offset_target[0], offset_target[1]))
                    session.set_target(target)
                    if not kat.dry_run:
                        session.ants.req.offset_fixed(offset_target[0], offset_target[1], opts.projection)
                    nd_params = session.nd_params
                    session.fire_noise_diode(announce=True, **nd_params)
                    if kat.dry_run:
                        session.track(target, duration=opts.track_duration, announce=False)
                    else:
                        time.sleep(opts.track_duration)  # Snooze

            # Tsys and averaging
            user_logger.info("Performing Tsys and averaging tests")
            session.nd_params = nd_off
            session.track(target, duration=0)  # get onto the source
            user_logger.info("Now capturing data - diode %s on" % nd_on['diode'])
            session.label('%s' % (nd_on['diode'],))
            if not session.fire_noise_diode(announce=True, **nd_on):
                user_logger.error("Noise Diode did not Fire , (%s did not fire)" % nd_on['diode'])
            session.nd_params = nd_off
            user_logger.info("Now capturing data - noise diode off")
            session.track(target, duration=300) # get 5 mins of data to test averaging

            # Single dish pointing ... to compare with interferometric
            user_logger.info("Performing single dish pointing tests")
            # RvR -- Very bad hack to keep from tracking above 89deg until AR1 AP can handle out of range values better
            if bad_ar1_alt_hack(target, 60.):
                print 'Too high elevation, skipping target %s...' % target.name
                continue
            # RvR -- Very bad hack to keep from tracking above 89deg until AR1 AP can handle out of range values better

            session.label('raster')
            user_logger.info("Doing scan of '%s' with current azel (%s,%s) " %
                             (target.description, target.azel()[0], target.azel()[1]))
            # Do different raster scan on strong and weak targets
            session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=6.0,
                                scan_spacing=0.25, scan_in_azimuth=True,
                                projection=opts.projection)
                        
            # reset the gains always            
            user_logger.info("Resetting F-engine gains to %g" % (opts.default_gain,))
            for inp in inputs:
                session.data.req.cbf_gain(inp, opts.default_gain)
