#!/usr/bin/env python
# Drive antenna in test pattern during encoder measurements

import time

from csv import reader
from katcorelib import (standard_script_options,
                        verify_and_connect,
                        user_logger)

ap_az = 0
ap_el = 15
ap_band = 's'
ap_delay = 0
ap_repeat = 1
ap_move_antenna = False
ap_move_band = False
ap_repeat_start = False
ap_repeat_stop = False
ap_skip = False
repeat_list = []
instruction_list = []


def stop_ants(kat):
    if not kat.dry_run and kat.ants.req.mode('STOP'):
        user_logger.info("Setting Antenna Mode to 'STOP', "
                         "Powering on Antenna Drives.")
        kat.ants.wait('mode', 'STOP', timeout=15)
    else:
        if not kat.dry_run:
            user_logger.error("Unable to set Antenna mode to 'STOP'.")


def move_antenna(kat, azim, elev):
    global ap_el
    global ap_az

    if (-185.0 > azim) or (azim > 275.0):
        raise ValueError("Cannot perform requested slew, azimuth travel range exceeded.")

    if (14.65 > elev) or (elev > 92.0):
        raise ValueError("Cannot perform requested slew, "
                         "elevation travel range exceeded.")

    # Use slew to bypass pointing model
    kat.ants.req.ap_slew(azim, elev)
    # Since we are not using the proxy request we will have to explicitly
    # wait for the servo brakes to open and on-target sensor to update.
    time.sleep(4)

    try:
        kat.ants.wait('ap.on-target', True, timeout=300)
        time.sleep(2)  # Track for a bit to allow deacceleration
        ap_el = int(elev)
        ap_az = int(azim)
    except Exception:
        user_logger.error("Timed out while waiting for AP to reach starting position.")
        raise

    user_logger.info("AP has reached start position.")

    stop_ants(kat)


def move_band(kat, band):
    if band not in ['u', 'l', 's', 'x']:
        raise ValueError("indexer position {} does not exist. active bands are 'u', 'l',"
                         "'s', and 'x'".format(band))
    user_logger.info("Moving receiver indexers to '{}' position".format(band))
    kat.ants.set_sampling_strategy("ap.ridx-brakes-released", "period 0.5")
    kat.ants.req.ap_set_indexer_position(band)
    time.sleep(10)
    if not kat.dry_run:
        try:
            # Wait for indexer brakes to engage again
            kat.ants.wait('ap.ridx-brakes-released', False, timeout=60)
            user_logger.info('{} position reached'.format(band))
        except Exception:
            not_on_pos = []
            not_on_pos = [ant.name for ant in kat.ants
                          if ant.sensor.ap_indexer_position.get_value() != band]
            user_logger.error("Indexer brakes did not engage on: {}"
                              .format(', '.join(not_on_pos)))
        finally:
            # allow brakes to engage and put antennas to stop
            time.sleep(3)
            stop_ants(kat)
            time.sleep(2)


def process_band(band):
    global ap_band
    global ap_move_band

    # We always want to move the indexer, even if it is already in 
    # the correct location for testing purposes
    ap_band = band
    ap_move_band = True


def process_az(az):
    global ap_az
    global ap_move_antenna
    if ap_az != int(az):
        ap_az = int(az)
        ap_move_antenna = True
    else:
        user_logger.info('Antenna already at Az: %d degrees', ap_az)


def process_el(el):
    global ap_el
    global ap_move_antenna
    if ap_el != int(el):
        ap_el = int(el)
        ap_move_antenna = True
    else:
        user_logger.info('Antenna already at El: %d degrees', ap_el)


def process_delay(delay):
    global ap_delay
    ap_delay = int(delay)


def process_repeat(repeat):
    global ap_repeat
    global ap_repeat_start
    global ap_repeat_stop
    global ap_skip
    if int(repeat) == 0:
        ap_skip = True
    elif ap_repeat != int(repeat):
        if ap_repeat_start:
            ap_repeat_start = False
            ap_repeat_stop = True
        else:
            ap_repeat_start = True

        ap_repeat = int(repeat)


def index_to_function(argument):
    switcher = {
        0: process_band,
        1: process_az,
        2: process_el,
        3: process_delay,
        4: process_repeat
    }
    return switcher.get(argument)


def parse_csv(csv_file):
    with open(csv_file, 'r') as f:
        steps = list(reader(f, delimiter=','))

    global ap_skip
    global ap_repeat
    global ap_repeat_start
    global ap_repeat_stop
    global repeat_list
    global instruction_list
    for step in range(1, len(steps)):
        # repeat is index 4
        if steps[step][4]:
            process_repeat(steps[step][4])
            if not ap_skip:
                if ap_repeat_start:
                    repeat_list.append(steps[step])
                elif not ap_repeat_stop:
                    instruction_list.append(steps[step])

                if ap_repeat_stop:
                    ap_repeat_stop = False
                    # copy the repeat list into the instruction list
                    for i in range(int(repeat_list[0][4])):
                        instruction_list.extend(repeat_list)
                    repeat_list = []

                    if ap_repeat != 1:
                        repeat_list.append(steps[step])
                        ap_repeat_start = True
                    else:
                        # new
                        instruction_list.append(steps[step])

            else:
                ap_skip = False

    if ap_repeat_start:
        # copy the repeat list into the instruction list
        for i in range(int(repeat_list[0][4])):
            instruction_list.extend(repeat_list)
        repeat_list = []
        ap_repeat_start = False

    print('Instruction list:')
    print(instruction_list)


def execute_instructions(kat, steps, dry_run=False):
    global ap_move_antenna
    global ap_move_band
    global ap_az
    global ap_el
    global ap_delay
    global ap_band

    if not dry_run:
        user_logger.info("Moving AP to starting position: Az: %d El: %d Band: %s",
                         ap_az, ap_el, ap_band)
        move_antenna(kat, float(ap_az), float(ap_el))
        move_band(kat, ap_band)

        for step in range(len(steps)):
            instruction = ('Az: ' + str(steps[step][1]) + ' El: ' + str(steps[step][2]) +
                ' Band: ' + str(steps[step][0]) + ' Delay: ' + str(steps[step][3]))
            user_logger.info('**** Executing instruction %d out of %d: %s ****',
                             step+1, len(steps), instruction)
            # skipping the repeat value in the steps list
            for action in range(4):
                if steps[step][action]:
                    index_to_function(action)(steps[step][action])

            if ap_move_antenna:
                ap_move_antenna = False
                move_antenna(kat, float(ap_az), float(ap_el))
            if ap_move_band:
                ap_move_band = False
                move_band(kat, ap_band)
            if ap_delay:
                user_logger.info('Sleeping for %d seconds', ap_delay)
                time.sleep(ap_delay)
                # Reset delay to 0 when we are done
                ap_delay = 0

        user_logger.info("Elevation pattern complete")


# Set up standard script options
parser = standard_script_options(
    usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
    description="Do basic full range elevation and azimuth movement for encoder "
    "count measurements")

parser.add_option('--csv-file', type='str',
                  default=None,
                  help='path to the csv file to parse')

# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

    # Set sensor strategies"
    kat.ants.set_sampling_strategy("ap.on-target", "event")
    kat.ants.set_sampling_strategy("ap.indexer-position", "event")
    kat.ants.set_sampling_strategy("mode", "event")

    stop_ants(kat)

    try:
        parse_csv(opts.csv_file)
        execute_instructions(kat, instruction_list, dry_run=kat.dry_run)
    finally:
        kat.ants.req.mode('STOP')
        user_logger.info("Stopping antennas")
