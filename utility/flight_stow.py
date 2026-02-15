#!/usr/bin/python
# Script to lower antennas in preparation for flight arrivals and departures from site

from katcorelib import (standard_script_options, verify_and_connect, user_logger)
import time
import katpoint
import numpy as np

def start_ants(ants, dry_run=False):
    if not dry_run:
        user_logger.info("Setting Antennae Mode to 'STOP', Powering on Antenna Drives.")
        ants.set_sampling_strategy('mode', 'event')
        ants.req.mode('STOP')
        try:
            ants.wait('mode', 'STOP', timeout=30) # 12 for MK, 30 for MKE
        except Exception as err:
            #user_logger.warn(err)	
            user_logger.warn("not all antennas were in 'STOP' mode.")

def move_ri(ants, ridx_pos, dry_run=False):
    if ridx_pos not in ['u', 'l', 's', 'x']:
        raise ValueError("indexer position {} does not exist. active bands are 'u', 'l', 's', and 'x'".format(ridx_pos))
    user_logger.info("Moving receiver indexers to '{}' position".format(ridx_pos))
    ants.set_sampling_strategy("ap.ridx-brakes-released","period 0.5")
    ants.req.ap_set_indexer_position(ridx_pos)
    time.sleep(10)
    if not dry_run:
        try:
            # Wait for indexer brakes to engage again
            ants.wait('ap.ridx-brakes-released', False, timeout=60)
            user_logger.info('{} position reached'.format(ridx_pos))
        except Exception:
            not_on_pos = []
            not_on_pos = [ant.name for ant in ants if ant.sensor.ap_indexer_position.get_value() != ridx_pos]
            user_logger.error("Indexer brakes did not engage on: {}".format(', '.join(not_on_pos)))
        finally:
            # allow brakes to engage and put antennas to stop
            time.sleep(3)
            ants.req.mode('STOP')
            time.sleep(2)

def point_ants(ants, dry_run=False):
    airstrip_lla = (np.radians(-30.690197222222224), np.radians(21.454725), 1080) # lat [rad], long [rad], alt [m]
    airstrip_heading = np.radians(123) # Azimuth angle, 0=north
    ant_ecef = [katpoint.Antenna(ant.sensor.observer.get_value()).position_ecef for ant in ants]
    ant_enu = [katpoint.ecef_to_enu(*(list(airstrip_lla)+list(xyz))) for xyz in ant_ecef]
    ant_headings = [np.arctan2(e,n) for e,n,u in ant_enu]
    user_logger.info("Airstrip is oriented along azim %gdeg" % np.degrees(airstrip_heading))
    
    ants.set_sampling_strategy('lock', 'event')
    for ant,heading in zip(ants, ant_headings):
        if (airstrip_heading-np.pi < heading < airstrip_heading): # North of airstrip, point NE
            az, el = (30, 18)
        else: # South of airstrip, point SW
            az, el = (217, 18)
        user_logger.info('%s heading is %.fdeg, slewing to azim: %g,   elev: %g' % (ant.name, np.degrees(heading), az, el))
        ant.req.target_azel(az, el)
        ant.req.mode('POINT')

    if not dry_run:
        try:
            ants.wait('lock', True, timeout=300)
            user_logger.info('target reached')
        except Exception:
            not_locked = []
            not_locked = [ant.name for ant in ants if ant.sensor.lock.get_value() == False]
            user_logger.warn('{} did not reach target'.format(', '.join(not_locked)))
        time.sleep(3)
        ants.req.mode('STOP')

# Set up standard script options
usage = '%prog'
description = 'receptor flight stow'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('--ridx-pos',
                  type="string",
                  default='x',
                  help="move receiver indexers to specified position. default=%default")
parser.add_option('--move-indexer',
                  action="store_true",
                  default=False,
                  help="enables the 'move receiver' routine. default=%default")
parser.set_defaults(observer='Tiyani', description='Receptor flight stow')
# Parse the command line
opts, args = parser.parse_args()

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    user_logger.info('Receptor flight stow START')
    # start antenna drives
    start_ants(kat.ants, dry_run=kat.dry_run)

    # point antennas
    if opts.move_indexer:
        move_ri(kat.ants, opts.ridx_pos, dry_run=kat.dry_run)
    point_ants(kat.ants, dry_run=kat.dry_run)
    user_logger.info('Receptor flight stow completed')
