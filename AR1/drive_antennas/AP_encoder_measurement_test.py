#!/usr/bin/env python
#Drive antenna in test pattern during encoder measurements

import time

from katcorelib import (standard_script_options,
                        verify_and_connect,
                        user_logger)


def move_to(ants, az, el):
    user_logger.info("Slewing to Az: '%s' El: '%s'", az, el)
    # Use slew to bypass pointing model
    ants.req.ap_slew(az, el)
    # Since we are not using the proxy mode we will have to explicitly wait
    # for the servo brakes to open and on-target sensor to update.
    time.sleep(4)

    try:
        ants.wait('ap.on-target', True, timeout=300)
        time.sleep(2)  # Track for a bit to allow deacceleration
        user_logger.info("Position reached...")
    except:
        user_logger.error("Timed out while waiting for AP to reach "
                          "requested position.")
        raise

    # Arbitrary pause before next movement
    time.sleep(10)


def elevation_test(ants, dry_run=False):

    user_logger.info("Elevation sequence.")
    if not dry_run:
        user_logger.info("Starting position...")
        move_to(ants, 0.0, 15.0)

        user_logger.info("High elevation (88 degrees)...")
        move_to(ants, 0.0, 88.0)

        user_logger.info("Low elevation (15 degrees)...")
        move_to(ants, 0.0, 15.0)

        user_logger.info("High elevation (88 degrees)...")
        move_to(ants, 0.0, 88.0)

        user_logger.info("Low elevation (15 degrees)...")
        move_to(ants, 0.0, 15.0)

        user_logger.info("Elevation pattern complete")


def azimuth_test(ants, dry_run=False):

    user_logger.info("Azimuth sequence.")
    if not dry_run:
        user_logger.info("Starting position...")
        move_to(ants, -185.0, 25.0)

        user_logger.info("CW limit (275 degrees)...")
        move_to(ants, 275.0, 25.0)

        user_logger.info("CCW limit (-185 degrees)...")
        move_to(ants, -185.0, 25.0)

        user_logger.info("CW limit (275 degrees)...")
        move_to(ants, 275.0, 25.0)

        user_logger.info("CCW limit (-185 degrees)...")
        move_to(ants, -185.0, 25.0)

        user_logger.info("Azimuth pattern complete")


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Do basic full range elevation and azimuth movement for encoder count measurements")

# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

    # Set sensor strategies"
    kat.ants.set_sampling_strategy("ap.on-target", "event")
    kat.ants.set_sampling_strategy("mode", "event")

    if not kat.dry_run and kat.ants.req.mode('STOP'):
        user_logger.info("Setting Antenna Mode to 'STOP', "
                         "Powering on Antenna Drives.")
        kat.ants.wait('mode', 'STOP', timeout=15)
    else:
        if not kat.dry_run:
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

    # Move to starting position for elevation stage of test
    elevation_test(kat.ants, dry_run=kat.dry_run)

    azimuth_test(kat.ants, dry_run=kat.dry_run)
