#!/usr/bin/python
# Script to lower antennas in preparation for a flight arrival or departure from site

from katcorelib import (standard_script_options, verify_and_connect, user_logger)
import time

def startAnts(ants, dry_run=False):
    if not dry_run:
	user_logger.info("Setting Antennae Mode to 'STOP', Powering on Antenna Drives.")
	ants.set_sampling_strategy('mode', 'event')
        ants.req.mode('STOP')
        try:
            ants.wait('mode', 'STOP', timeout=12)
        except:
            user_logger.warn("not all antennas were in 'STOP' mode.")

def point_ants(ants, dry_run=False):
    ant_diff = ['m057', 'm058', 'm059']
    for ant in ants:
	if ant.name in ant_diff:
	    user_logger.info('slewing %s to az: 30, el: 18' % ant.name)
	    ant.req.target_azel(30, 18)
	else:
	    user_logger.info('slewing %s to az: 217, el: 18' % (ant.name))
	    ant.req.target_azel(217, 18)
	ant.req.mode('POINT')

    if not dry_run:
	try:
	    ants.wait('lock', True, timeout=300)
	    user_logger.info('target reached')
	except:
	    not_locked = []
	    not_locked = [ant.name for ant in ants if ant.sensor.lock.get_value() == False]
	    user_logger.warn('{} did not reach target'.format(not_locked))
	time.sleep(5)
	ants.req.mode('STOP')

# Set up standard script options
usage = '%prog'
description = 'receptor flight stow'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.set_defaults(observer='Tiyani', description='Receptor flight stow')
# Parse the command line
opts, args = parser.parse_args()

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    user_logger.info('receptor flight stow START')
    # start antenna drives
    startAnts(kat.ants, dry_run=kat.dry_run)

    # point antennas
    kat.ants.set_sampling_strategy('lock', 'event')
    point_ants(kat.ants, dry_run=kat.dry_run)
    user_logger.info('receptor flight stow completed')
