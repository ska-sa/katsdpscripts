#!/usr/bin/env python
# Exercise the indexer at various elevations.

import time
import katpoint

from katcorelib import (standard_script_options,
                        verify_and_connect,
                        user_logger)


class ApDriveFailure(Exception):
    """AP failed to move due to drive failure."""


class UndefinedPosition(Exception):
    """Indexer is stuck in an undefined position."""


def stow_test(ants, taz, tel, dry_run=False):

    # send this target to the antenna.
    target = katpoint.Target('Name, azel, %s, %s' % (taz, tel))

    if not dry_run:
        # Use slew to bypass pointing model
        ants.req.ap_slew(taz, tel)
        # Since we are not using the proxy mode we will have to explicitly wait
        # for the servo brakes to open and on-target sensor to update.
        time.sleep(4)

    user_logger.info("Starting target description: '%s' ", target.description)

    if not dry_run:
        try:
            ants.wait('ap.on-target', True, timeout=300)
            time.sleep(2)  # Track for a bit to allow deacceleration
        except:
            user_logger.error("Timed out while waiting for AP to reach "
                              "starting position.")
            raise

        user_logger.info("AP has reached start position, beginning stow test")

        user_logger.info("Activating receptor windstows")
        ants.req.set_windstow(1)
        ants.wait('ap.mode', 'stowed', timeout=120)

        # Wait a bit before clearing
        user_logger.info("Stow position reached. Monitoring for 20 seconds.")
        time.sleep(20)
        user_logger.info("Clearing receptor windstow condition")
        ants.req.set_windstow(0)
        ants.wait('mode', 'STOP', timeout=120)

        # Wait a bit and issue the windstow again to simulate wind picking up
        user_logger.info("Waiting 30 seconds before next 'gust'.")
        time.sleep(30)
        user_logger.info("Triggering 2nd windstow event.")
        ants.req.set_windstow(1)
        ants.wait('mode', 'STOW', timeout=120)

        user_logger.info("Montitoring for 20 seconds")
        # Reduced this time since we should already be up at stow position
        time.sleep(20)

        user_logger.info("Test concluded. Clearing windstow.")
        # Restore system to normal
        ants.req.set_windstow(0)
        ants.wait('mode', 'STOP', timeout=120)


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Exercise the receiver indexer drive at different elevations.")

parser.add_option('--start-az', type='float',
                  default=50.0,
                  help='Starting azimuth (default=%default)')
parser.add_option('--start-el', type='float',
                  default=50.0,
                  help='Starting elevation (default=%default)')

# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

    # Set sensor strategies"
    kat.ants.set_sampling_strategy("lock", "event")
    kat.ants.set_sampling_strategy("ap.on-target", "event")
    kat.ants.set_sampling_strategy("mode", "event")
    kat.ants.set_sampling_strategy("mode", "event")
    kat.ants.set_sampling_strategy("ap.mode", "event")

    if not kat.dry_run and kat.ants.req.mode('STOP'):
        user_logger.info("Setting Antenna Mode to 'STOP', "
                         "Powering on Antenna Drives.")
        kat.ants.wait('mode', 'STOP', timeout=15)
    else:
        if not kat.dry_run:
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

    stow_test(kat.ants, float(opts.start_az), float(opts.start_el),
              dry_run=kat.dry_run)
