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
    """Requested position is outside the allowed travel range."""


def move_antennas(ants, azim, elev):
    if (-185.0 > azim) or (azim > 275.0):
        raise ParametersExceedTravelRange("Cannot perform requested slew, "
                                          "azimuth travel range exceeded.")

    if (14.65 > elev) or (elev > 92.0):
        raise ParametersExceedTravelRange("Cannot perform requested slew, "
                                          "elevation travel range exceeded.")

    # Use slew to bypass pointing model
    ants.req.ap_slew(azim, elev)
    # Since we are not using the proxy request we will have to explicitly
    # wait for the servo brakes to open and on-target sensor to update.
    time.sleep(4)

    try:
        ants.wait('ap.on-target', True, timeout=300)
        time.sleep(2)  # Track for a bit to allow deacceleration
    except:
        user_logger.error("Timed out while waiting for AP to reach starting position.")
        raise

    user_logger.info("AP has reached start position.")

    ants.req.mode('STOP')
    time.sleep(3)

    current_az = ants.req.sensor_value('ap.actual-azim')
    current_el = ants.req.sensor_value('ap.actual-elev')
    current_ridx = ants.req.sensor_value('ap.indexer-position-raw')

    user_logger.info("Azimuth readings: %s", current_az)
    user_logger.info("Elevation readings: %s", current_el)
    user_logger.info("Indexer position readings: %s", current_ridx)


def test_sequence(ants, disable_corrections=False, dry_run=False):

    # Disable ACU pointing corrections
    if not dry_run:
        if disable_corrections:
            SPEM_state = False
            TILT_state = True
            ants.req.ap_enable_point_error_systematic(False)
            ants.req.ap_enable_point_error_tiltmeter(False)
            user_logger.warning("ACU pointing corrections have been disabled.")

    for azim in [-45, 45, 135, 225]:

        # Start at low elevation
        elev = 20
        # Set this target for the receptor target sensor
        target = katpoint.Target('Name, azel, %s, %s' % (azim, elev))
        user_logger.info("Starting position: '%s' ", target.description)

        if not dry_run:
            move_antennas(ants, azim, elev)

        # Move to high elevation
        elev = 80
        # Set this target for the receptor target sensor
        target = katpoint.Target('Name, azel, %s, %s' % (azim, elev))
        user_logger.info("Move to high elevation: '%s' ", target.description)

        if not dry_run:
            move_antennas(ants, azim, elev)
            # Wait for 10 minutes before moving to next position
            time.sleep(600)

        # Return to low elevation
        elev = 20
        # Set this target for the receptor target sensor
        target = katpoint.Target('Name, azel, %s, %s' % (azim, elev))
        user_logger.info("Return to low elevation: '%s' ", target.description)

        if not dry_run:
            move_antennas(ants, azim, elev)

    # Restore ACU pointing corrections
    if not dry_run:
        if disable_corrections:
            ants.req.ap_enable_point_error_systematic(SPEM_state)
            ants.req.ap_enable_point_error_tiltmeter(TILT_state)

    user_logger.info("Sequence Completed!")


# Set up standard script options

parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Slew the antennas in elevation at different azimuth.")

parser.add_option('--no-corrections', action='store_true',
                  help='Disable static and tilt corrections during the controlled movement, restore afterwards.')


# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

    # Set sensor strategies"
    kat.ants.set_sampling_strategy("ap.on-target", "event")

    if not kat.dry_run and kat.ants.req.mode('STOP'):
        user_logger.info("Setting antennas to mode 'STOP'")
        time.sleep(2)
    else:
        if not kat.dry_run:
            raise RuntimeError("Unable to set antennas to mode 'STOP'!")

    try:
        test_sequence(kat.ants,
                      disable_corrections=opts.no_corrections,
                      dry_run=kat.dry_run)
    finally:
        if not kat.dry_run:
            kat.ants.req.mode('STOP')
            user_logger.info("Stopping antennas")
