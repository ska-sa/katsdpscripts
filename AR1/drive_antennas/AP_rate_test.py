#!/usr/bin/env python
# Exercise the indexer at various elevations.

import time
import katpoint

from katcorelib import (standard_script_options,
                        verify_and_connect,
                        user_logger)


class ApDriveFailure(Exception):
    """AP failed to move due to drive failure."""


class ParametersExceedTravelRange(Exception):
    """Indexer is stuck in an undefined position."""


def rate_slew(ants, azim, elev, speed=0.5, reverse=False, disable_corrections=False, dry_run=False):

    # Scale timeout with requested test speed. Full speed timeout is 5 min
    rate_timeout = (float(2)/speed)*300
    # Only testing movement in azimuth, not elevation
    azim_speed = speed
    elev_speed = 0.0
    sensor_name = "ap.actual-azim"
    # Only testing 365 degrees, not full travel range (460 degrees)
    expected_azim = (azim + 365.0 if speed > 0 else azim - 365.0)
    # Position threshold 2 degrees to catch it at 0.5 second polling
    # period at full speed (2 deg/sec).
    threshold = 2

    if (-185.0 < expected_azim < 275.0):
        user_logger.info("Antennas will perform a rate slew to azimuth %s",
                         expected_azim)
    else:
        raise ParametersExceedTravelRange("Cannot perform 365 degree slew "
                                          "within the AP azim travel range "
                                          "from the given start position.")

    # Set this target for the receptor target sensor
    target = katpoint.Target('Name, azel, %s, %s' % (azim, elev))

    if not dry_run:
        # Use slew to bypass pointing model
        ants.req.ap_slew(azim, elev)
        # Since we are not using the proxy request we will have to explicitly
        # wait for the servo brakes to open and on-target sensor to update.
        time.sleep(4)

    user_logger.info("Starting target description: '%s' ", target.description)

    if not dry_run:
        try:
            ants.wait('ap.on-target', True, timeout=300)
            time.sleep(2)  # Track for a bit to allow deacceleration
        except:
            user_logger.error("Timed out while waiting for AP to reach starting position.")
            raise

    user_logger.info("AP has reached start position.")

    if not dry_run:
        if disable_corrections:
            SPEM_state = False # np.any([a.sensors.ap_point_error_systematic_enabled.get_value() for a in ants]) # TODO finalize the commented-out snippet
            TILT_state = True # np.any([a.sensors.ap_point_error_tiltmeter_enabled.get_value() for a in ants])
            ants.req.ap_enable_point_error_systematic(False)
            ants.req.ap_enable_point_error_tiltmeter(False)
        
        ants.req.mode('STOP')
        time.sleep(3)

        ants.req.ap_rate(azim_speed, elev_speed)
        user_logger.info("Performing rate slew to azimuth %s at %s deg/sec.",
                         expected_azim, azim_speed)
        # Wait until we are within a threshold of the expected ending azimuth
        ants.wait(sensor_name,
                  lambda c: abs(c.value - expected_azim) < threshold,
                  timeout=rate_timeout)

        if reverse:
            user_logger.info("Reverse slew selected...")
            ants.req.mode('STOP')
            time.sleep(8)
            ants.req.ap_rate(-1*azim_speed, elev_speed)
            user_logger.info("Performing rate slew to azimuth %s at %s deg/sec.",
                             azim, azim_speed)
            # Head back to the starting azim that was passed into the function
            ants.wait(sensor_name,
                      lambda c: abs(c.value - azim) < threshold,
                      timeout=rate_timeout)

        if disable_corrections:
            ants.req.ap_enable_point_error_systematic(SPEM_state)
            ants.req.ap_enable_point_error_tiltmeter(TILT_state)

    user_logger.info("Sequence Completed!")


# Set up standard script options

parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Slew the antennas in azimuth at the specified speed.")

parser.add_option('--start-az', type='float',
                  default=-135.0,
                  help='Starting azimuth for slew sequence (default=%default)')
parser.add_option('--start-el', type='float',
                  default=20.0,
                  help='Starting elevation for slew sequence (default=%default)')
parser.add_option('--azim-speed', type='float',
                  default=0.5,
                  help='Azimuth slew speed in deg/sec (default=%default)')
parser.add_option('--reverse', action='store_true', default=False,
                  help='Do the rate movement in both directions')
parser.add_option('--no-corrections', action='store_true',
                  help='Disable static and tilt corrections during the controlled movement, restore afterwards.')


# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

    # Set sensor strategies"
    kat.ants.set_sampling_strategy("ap.actual-azim", "period 0.5")
    kat.ants.set_sampling_strategy("ap.on-target", "event")

    if not kat.dry_run and kat.ants.req.mode('STOP'):
        user_logger.info("Setting antennas to mode 'STOP'")
        time.sleep(2)
    else:
        if not kat.dry_run:
            raise RuntimeError("Unable to set antennas to mode 'STOP'!")

    receptors = kat.ants

    try:
        rate_slew(receptors, float(opts.start_az), float(opts.start_el),
                  speed=float(opts.azim_speed), reverse=opts.reverse,
                  disable_corrections=opts.no_corrections, dry_run=kat.dry_run)
    finally:
        if not kat.dry_run:
            kat.ants.req.mode('STOP')
            user_logger.info("Stopping antennas")
