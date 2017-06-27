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


def wait_until_sensor_equals(timeout, sensorname, value,
                             sensortype=str, places=5, pollperiod=0.5):

    stoptime = time.time() + timeout
    success = False

    if sensortype == float:
        cmpfun = lambda got, exp: abs(got - exp) < 10 ** -places
    else:
        cmpfun = lambda got, exp: got == exp

    lastval = None
    while time.time() < stoptime:
        lastval = kat.sensors.get(sensorname, None).get_value()

        if cmpfun(lastval, value):
            success = True
            break
        time.sleep(pollperiod)
    return (success, lastval)


def track(ant, taz, tel, ridx_position, duration=10, total=1, dry_run=False):
  
    # Direction variable ie. up/down, clockwise/anti-clockwise
    h_direction = 1
    v_direction = 1

    # Number of cycles done
    cycle_count = 0
    
    # Total number of degrees travelled in each axis
    az_total_angle = 0
    el_total_angle = 0
     
    #Total number of each type of slew completed
    az_7_deg_slews = 0
    el_7_deg_slews = 0
    az_26_deg_slews = 0
    el_23_deg_slews = 0
    
    # send this target to the antenna.
    target = katpoint.Target('Name, azel, %s, %s' % (taz, tel))
    
    if not dry_run:
        # Use slew to bypass pointing model
        ant.req.ap_slew(taz, tel)
        # Since we are not using the proxy mode we will have to explicitly wait
        # for the servo brakes to open and on-target sensor to update.
        time.sleep(4)

    user_logger.info("Starting target description: '%s' ", target.description)

    if not dry_run:
        try:
            ant.wait('ap.on-target', True, timeout=300)
            time.sleep(2) # Track for a bit to allow deacceleration
        except:
            user_logger.error("Timed out while waiting for AP to reach starting position.")
            raise
   
    user_logger.info("AP has reached start position, beginning endurance cycle")
    last_az = ant.sensor.ap_actual_azim.get_value()
    last_el = ant.sensor.ap_actual_elev.get_value()
    
    while cycle_count <= total:
        # Cycle loop
        # Azimuth cycle is 7, 26, 7 degrees
        # Elevation cycle is 7, 23.5, 7 degrees
        # Azel = [(7,7),(26,23.5),(7,7)]
        for az_offset, el_offset in [(7,7),(26,23),(7,7)]:
            taz += az_offset*h_direction  # Find new coordinate based on current position
            tel += el_offset*v_direction

            # send this target to the antenna.
	    target = katpoint.Target('Name, azel, %s, %s' % (taz, tel))

            if not dry_run:
                # Use slew to bypass pointing model
                ant.req.ap_slew(taz, tel)
                # Since we are not using the proxy mode we will have to explicitly wait
                # for the servo brakes to open and on-target sensor to update.
                time.sleep(4)

            user_logger.info("Target description: '%s' ( %s degree slew )", target.description, str(az_offset))

            if not dry_run:
                try:
                    ant.wait('ap.on-target', True, timeout=300)
                    time.sleep(2) # Track for a bit to allow deacceleration
                except:
                    user_logger.error("Timed out while waiting for AP to complete a slew.")
                    user_logger.info("7 deg slews: az - '%s', el - '%s' ", az_7_deg_slews, el_7_deg_slews)
                    user_logger.info("26/23 deg slews: az - '%s', el - '%s' ", az_26_deg_slews, el_23_deg_slews)
                    user_logger.info("Total degrees travelled: az - '%s', el - '%s' ", az_total_angle, el_total_angle)
                    raise

            # Get the current position
            current_az = ant.sensor.ap_actual_azim.get_value()
            current_el = ant.sensor.ap_actual_elev.get_value()
            
            # Add the angle travelled to the accumulated value
            az_total_angle += abs(current_az - last_az)
            el_total_angle += abs(current_el - last_el)
            
            # Update last_<axis> values
            last_az = current_az
            last_el = current_el
            
            if az_offset == 7:
                az_7_deg_slews += 1
            else:
                az_26_deg_slews += 1
            if el_offset == 7:
               el_7_deg_slews += 1
            else:
                el_23_deg_slews += 1
        
        # Update cycle counter
        cycle_count += 1
        # If we have done 2 elevation cycles then reverse direction
        if cycle_count % 2 == 0:
            v_direction = v_direction*-1
        # If we have done 9 azimuth cycles then reverse direction
        if cycle_count % 9 == 0:
            h_direction = h_direction*-1

        # Only need this bit if we are not doing indexer test
        #if not dry_run:
        #   time.sleep(228)

        ant.req.mode('STOP')
        # Add a sleep since the indexer portion of the script does not
        # use the waiting functionality that is part of the receptor
        # proxy mode requests
        if not dry_run:
           time.sleep(3)

        indexer_timeout = 120
        # Position raw changed after indexer configurations
        ridx_angle={'s':-0.618,'l':39.248,'x':79.143,'u':119.405}
        ridx_sequence = ['s','l','x','u']

        # If we are closer to 'u' than 's', then start from 'u' instead
        if ant.sensor.ap_indexer_position_raw.get_value() > 60:
            ridx_sequence.reverse()

        ridx_last_position = ant.sensor.ap_indexer_position.get_value()

        # NOTE: we skip the first position in the list if we are already
        #       there because the indexer seems to end up in a 'moving'
        #       state after a few cycles of operating the drive. Revisit
        #       this after feedback from Nico
        if ridx_last_position == ridx_sequence[0]:
            ridx_sequencei = ridx_sequence[1:]

        if not dry_run:
            for pos in ridx_sequence:

                ridx_movement_start_time = time.time()
                user_logger.info("--- Moving RI to position: '%s' ---", pos.upper())
                ant.req.ap_set_indexer_position(pos)

                # Wait for indexer brakes to open
                time.sleep(2)
                try:
                    ant.wait('ap.ridx-brakes-released', False, timeout=60)
                except:
                    raise

                # Wait for power to encoder to switch off
                time.sleep(5)

                ridx_current_position = ant.sensors.ap_indexer_position.get_value()
                time_to_index = time.time() - ridx_movement_start_time

                ridx_position_raw = ant.sensor.ap_indexer_position_raw.get_value()
                ridx_brakes_released = ant.sensor.ap_ridx_brakes_released.get_value()
                if ridx_current_position != pos:

                    user_logger.error("Timed out while waiting %s seconds "
                                      "for indexer to reach '%s' position. "
                                      "Last position reading was %s degrees. "
                                      "Brakes released: '%s'. ",
                                      time_to_index, pos.upper(),
                                      ridx_position_raw,
                                      ridx_brakes_released)

                    user_logger.info("7 deg slews: az - '%s', el - '%s' ", az_7_deg_slews, el_7_deg_slews)
                    user_logger.info("26/23 deg slews: az - '%s', el - '%s' ", az_26_deg_slews, el_23_deg_slews)
                    user_logger.info("Total degrees travelled: az - '%s', el - '%s' ", az_total_angle, el_total_angle)

                    raise UndefinedPosition("Indexer failed to reach the requested position.") 
                else:
                    user_logger.info("Brake engaged. The offset from the requested position: "
                                     "'%s' is %.6f degree(s)",
			                         pos.upper(),
                                     abs(ridx_angle[pos]- ridx_position_raw))
                    user_logger.info("Request for angle '%s' to final angle '%s' "
                                     "took '%s' seconds.",
                                     ridx_angle[pos],
                                     ridx_position_raw,
                                     time_to_index)

                # Wait a little before requesting next indexer position
                time.sleep(5)
                   
    # Print out slew numbers once all cycles are completed 
    user_logger.info("7 deg slews: az - '%s', el - '%s' ", az_7_deg_slews, el_7_deg_slews)
    user_logger.info("26/23 deg slews: az - '%s', el - '%s' ", az_26_deg_slews, el_23_deg_slews)
    user_logger.info("Total degrees travelled: az - '%s', el - '%s' ", az_total_angle, el_total_angle)


# Set up standard script options

parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Exercise the receiver indexer drive at different elevations.")

parser.add_option('--dwell-time', type='float',
                  default=5.0,
                  help='Time between changing indexer positions in seconds (default=%default)')
parser.add_option('--num-repeat', type='int',
                  default=1,
                  help='The number of times to repeat the sequence (once by by default)')
parser.add_option('--ap', type='str',                                   
                  default="m036",                                                    
                  help='Receptor under test (default is m036)')

# Parse the command line
opts, args = parser.parse_args()

if opts.observer is None:
    raise RuntimeError("No observer provided script")

if len(args) == 0:
    raise RuntimeError("No targets and indexer positions"
                       "provided to the script")

receptor = None
for argstr in args:
    temp, taz, tel, band = argstr.split(',')
# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    
    for ant in kat.ants:
        if opts.ap in [ant.name]:
            receptor = ant

    if receptor is None:
        raise RuntimeError("Receptor under test is not in controlled array")

    # Set sensor strategies"
    kat.ants.set_sampling_strategy("lock", "event")
    kat.ants.set_sampling_strategy("ap.indexer-position", "period 0.1")
    kat.ants.set_sampling_strategy("ap.on-target", "event")
    kat.ants.set_sampling_strategy("ap.ridx-brakes-released","period 0.1")

    if not kat.dry_run and receptor.req.mode('STOP'):
        user_logger.info("Setting Antenna Mode to 'STOP', "
                         "Powering on Antenna Drives.")
        time.sleep(2)
    else:
        if not kat.dry_run:
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

 
    track(receptor, int(taz), int(tel), ridx_position=band, duration=10,
          total=int(opts.num_repeat), dry_run=kat.dry_run)
