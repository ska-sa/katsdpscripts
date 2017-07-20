#!/usr/bin/python
# Script to lower antennas in preparation for a flight arrival or departure from site

from katcorelib import (standard_script_options, verify_and_connect, user_logger)
from katcorelib import (cambuild,  katconf)
from random import randint
import time

def point_ants(ants, dry_run=False):
    ant_diff = ['m057', 'm058', 'm059']
    for ant in ants:
	if ant.name in ant_diff:
	    user_logger.info('slewing %s to az: 150, el: 18' % ant.name) 
	    ant.req.target_azel(150,18)
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
user_logger.info('start receptor flight stow\n')

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    user_logger.info('receptor flight stow begin')
    if not kat.dry_run and kat.ants.req.mode('STOP'):
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(5)
    else:
        if not kat.dry_run :
            user_logger.warn("Unable to set Antenna mode to 'STOP'.")
  
    # set AP sampling strategy
    kat.ants.set_sampling_strategy('lock', 'event')

    # point antennas
    point_ants(kat.ants, dry_run=kat.dry_run)
    user_logger.info('receptor flight stow completed')
