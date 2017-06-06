#!/usr/bin/python
# Silly script to check the antennas
#
# Initial script
# Quick check: run check_ant_AR1.py -o ruby
# Better check after technical maintenance: run check_ant_AR1.py -o ruby --tech
# Better check and reset all failure states: run check_ant_AR1.py -o ruby --tech --reset
# Tiyani adding sensors: Indexer pos, safe key switch, acu encoders
# will include receiver band selection to query the selected receiver and dititiser band later


from __future__ import with_statement
import time, string
from katcorelib import standard_script_options, verify_and_connect, user_logger, start_session
from katcorelib import cambuild, katconf

def check_sensors(ped, sensor_list, test_false=False):
    errors_found=False
    for atr in sensor_list:
	if getattr(ped, atr).get_value():
	    errors_found=True
	    if not test_false:
	        print '    Error detected: %s is %s' % (atr,getattr(ped, atr).get_value())
    return errors_found

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="AR1 antenna quick check")
parser.add_option('--ant', type=str, default='m063',
                  help='Antennas to check in the format m0xx. (default="%default")')
parser.add_option('--tech', action="store_true", default=False,
                  help='Extensive verification after antenna returned from technical maintenance')
parser.add_option('--reset', action="store_true", default=False,
                  help='Reset failure states on antennas before checking sensors')
# assume basic options passed from instruction_set
parser.set_defaults(description = 'AR1 AP Quick Check')
(opts, args) = parser.parse_args()
if opts.ant is None: 
    raise SystemExit('antenna name required %s' % parser.print_usage())


print("Antenna quick check : start")


with verify_and_connect(opts) as kat:
    try:
        cam = None
        done = False
        count = 1

        if not kat.dry_run:
	    print('Building CAM object')
            cam = cambuild(password="camcam", full_control="all")
	    time.sleep(5)

            # ant_active = [ant for ant in cam.ants if ant.name not in cam.katpool.sensor.resources_in_maintenance.get_value()]
	    # list all antennas so that you can test an antenna that is in maintenance
            ant_active = [ant for ant in cam.ants]
            for ant in ant_active:
	        if ant.name == opts.ant:
		    any_errors=False
		    # Verify that all the config data has been added from RTS
		    print 'Antenna %s is connected' % ant.name
		    print '  L-band serial number: %s' % ant.sensor.rsc_rxl_serial_number.get_value()
                    
                    # Checking the receiver index position.
                    # Note: not all indexers have predefined string positions
                    print '  checking receiver position. indexer value %s.' % ant.sensor.ap_indexer_position_raw.get_value()
                    if ant.sensor.ap_indexer_position_raw.get_value() > 59 and ant.sensor.ap_indexer_position_raw.get_value() < 62:
                        print '  Indexer position in Maintenance position'
                    elif ant.sensor.ap_indexer_position_raw.get_value() > 35 and ant.sensor.ap_indexer_position_raw.get_value() < 42:
                        print '  indexer position label is %s_band' % ant.sensor.ap_indexer_position.get_value().upper()
                    else:
                        print '  Index in unknown position'


		    if opts.tech:
		        if opts.reset: ant.req.ap_reset_failures()

		        if not ant.sensor.ap_connected.get_value():
		            raise RuntimeError('AP is not connected')
		        if not ant.sensor.comms_ok.get_value():
		            raise RuntimeError('AP comms is not okay')
		        if ant.sensor.ap_control.get_value() != 'remote':
		            raise RuntimeError('AP cannot be remotely controlled')
		        if ant.sensor.ap_e_stop_reason.get_value() != 'none':
		            raise RuntimeError('AP e_stop %s' % ant.sensor.ap_e_stop_reason.get_value())

                        print '  Checking receiver LNA power   '
                        if not ant.sensor.rsc_rxl_lna_h_power_enabled.get_value():
                            raise RuntimeError('receiver hpol power is OFF')
                        else:
                            print '  :) receiver hpol power is ON'
                        if not ant.sensor.rsc_rxl_lna_v_power_enabled.get_value():
                            raise RuntimeError('receiver vpol power is OFF')
                        else:
                            print '  :) receiver vpol power is ON'

                        print '  checking acu encoder'
                        enc=[
                                    'ap_azim_enc_failed',
                                    'ap_elev_enc_failed'
                            ]
                        if check_sensors(ant.sensor, enc):
                            any_errors = True
                        else:
                            print '  :) ACU encoders OK'

                        print '  checking antenna key lock switch'
                        safe_pos=[
                                    'ap_key_switch_safe1_enabled',
                                    'ap_key_switch_safe2_enabled'
                                 ]
                        if check_sensors(ant.sensor, safe_pos):
                            any_errors = True
                        else:
                            print '  :) antenna is unlocked'

		        print '  Checking motion'
		        amps=[
                                    'ap_azim_amp1_failed',
                                    'ap_azim_amp2_failed',
                                    'ap_azim_motion_error',
                                    'ap_azim_servo_failed',
                                    'ap_elev_amp_failed',
                                    'ap_elev_motion_error',
                                    'ap_elev_servo_failed',
                                    'ap_ridx_amp_failed'
                             ]
		        if check_sensors(ant.sensor, amps):
		            any_errors = True
                        else:
                            print '  :) AP Servo OK'

		        print '  Checking breaks'
		        breaks=[
                                    'ap_azim_brake1_failed',
                                    'ap_azim_brake2_failed',
                                    'ap_elev_brake_failed',
                                    'ap_ridx_brake_failed'
                               ]
		        if check_sensors(ant.sensor, breaks):
		            any_errors = True
                        else:
                            print '  :) Brakes OK'

		        breaks=[
                                    'ap_azim_brakes_released',
                                    'ap_elev_brakes_released'
                                   ]
		        if check_sensors(ant.sensor, breaks, test_false=True):
		            any_errors = True
                        else:
                            print '  :) Brakes released'
                            
		        print '  Checking doors'
		        doors=[
                                    'ap_hatch_door_open', 
                                    'ap_ped_door_open', 
                                    'ap_yoke_door_open'
                                  ]
		        if check_sensors(ant.sensor, doors):
		            any_errors = True
                        else:
                            print '  :) All Doors Closed'

		        print '  Checking limits'
		        azim_limits=[
                                    'ap_azim_emergency2_limit_ccw_reached',
                                    'ap_azim_emergency2_limit_cw_reached',
                                    'ap_azim_emergency_limit_ccw_reached',
                                    'ap_azim_emergency_limit_cw_reached',
                                    'ap_azim_hard_limit_ccw_reached',
                                    'ap_azim_hard_limit_cw_reached',
                                    'ap_azim_prelimit_ccw_reached',
                                    'ap_azim_prelimit_cw_reached',
                                    'ap_azim_soft_limit_ccw_reached',
                                    'ap_azim_soft_limit_cw_reached',
                                    'ap_azim_soft_prelimit_ccw_reached',
                                    'ap_azim_soft_prelimit_cw_reached'
                                    ]
		        if check_sensors(ant.sensor, azim_limits):
		            any_errors = True
                        else:
                            print '  :) AP not in Azimuth Limit'

		        elev_limits=[
		                    'ap_elev_emergency2_limit_down_reached',
		                    'ap_elev_emergency2_limit_up_reached',
		                    'ap_elev_emergency_limit_down_reached',
		                    'ap_elev_emergency_limit_up_reached',
		                    'ap_elev_hard_limit_down_reached',
		                    'ap_elev_hard_limit_up_reached',
		                    'ap_elev_soft_limit_down_reached',
		                    'ap_elev_soft_limit_up_reached',
		                    'ap_elev_soft_prelimit_down_reached',
		                    'ap_elev_soft_prelimit_up_reached'
                                    ]
		        if check_sensors(ant.sensor, elev_limits):
		            any_errors = True
                        else:
                            print '  :) AP not in Elevation Limit'

# 		        ridx_limits=[
# 		        'ap_ridx_hard_limit_ccw_reached',
# 		        'ap_ridx_hard_limit_cw_reached',
# 		        'ap_ridx_soft_limit_ccw_reached',
# 		        'ap_ridx_soft_limit_cw_reached',
# 		        'ap_ridx_soft_prelimit_ccw_reached',
# 		        'ap_ridx_soft_prelimit_cw_reached']
# 		        if check_sensors(ant.sensor, ridx_limits):
# 		            any_errors = True

#RvR -- feedback from EMSS is to only report of an LNA is off and not to try and switch it on for the moment
# 	    	    if not ant.sensor.rsc_rxl_amp2_h_power_enabled.get_value():
# 	    	        print 'Switching on L-band Amp2 H'
# 	    	        ant.req.rsc_rxl_amp2_h_power('enable')
# 	    	        time.sleep(1)
# 		    if not ant.sensor.rsc_rxl_amp2_v_power_enabled.get_value():
# 		        print 'Switching on L-band Amp2 V'
# 		        ant.req.rsc_rxl_amp2_v_power('enable')
# 		        time.sleep(1)
# 		    if not ant.sensor.rsc_rxl_lna_h_power_enabled.get_value():
# 		        print 'Switching on L-band LNA H'
# 		        ant.req.rsc_rxl_lna_h_power('enable')
# 		        time.sleep(1)
# 		    if not ant.sensor.rsc_rxl_lna_v_power_enabled.get_value():
# 		        print 'Switching on L-band LNA V'
# 		        ant.req.rsc_rxl_lna_v_power('enable')
# 		        time.sleep(1)
#	            print '  Checking LNAs'
#		    lna_status=[
#		    'rsc_rxl_amp2_h_power_enabled',
#		    'rsc_rxl_amp2_v_power_enabled',
#		    'rsc_rxl_lna_h_power_enabled',
#		    'rsc_rxl_lna_v_power_enabled']
#		    if not check_sensors(ant.sensor, lna_status, test_false=True):
#		        any_errors = True

		    if ant.sensor.rsc_rxl_startup_state.get_value() != 'cold-operational':
		        raise RuntimeError('rsc_rxl_startup_state is %s, cold-operational expected'% ant.sensor.rsc_rxl_startup_state.get_value())
		    if ant.sensors.rsc_rxl_rfe1_temperature.get_value() > 30:
		        raise RuntimeError('Antenna is warm -- rsc_rxl_rfe1_temperature is %.2f K'% ant.sensors.rsc_rxl_rfe1_temperature.get_value())

		    if any_errors:
		        print "Some errors detected, investigate before accepting %s\n" %ant.name
		    else:
		        print 'All tests passed and LNAs are on\n'

    finally:
        if cam:
	    print("Cleaning up cam object")
            cam.disconnect()

# -fin-

