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
	        user_logger.warning("Error detected: %s is %s" % (atr,getattr(ped, atr).get_value()))
    return errors_found

def check_digitisers():
    if ant.sensor.dig_selected_band.get_value() not in ["u", "l", "s", "x"]:
        user_logger.warning("digitiser is in %s band. expected u, l, s or x band" % ant.sensor.dig_selected_band.get_value())
    else:
        print("digitiser is in %s band" % ant.sensor.dig_selected_band.get_value())

# Checking L-band receiver for now. will add more checks for UHF and others on the next push
def check_receivers():
    if ant.sensor.rsc_rxl_startup_state.get_value() != "cold-operational":
        raise RuntimeError("rsc_rxl_startup_state is %s, cold-operational expected"% ant.sensor.rsc_rxl_startup_state.get_value())
    
    if ant.sensors.rsc_rxl_rfe1_temperature.get_value() < 29: 
        print("L-band rfe1 temperature is ok :)") 
        if not ant.sensor.rsc_rxl_lna_h_power_enabled.get_value():
            user_logger.warning("L-band receiver hpol LNA power is not enabled. switch on the hpol LNA power")
        else:
            print(":) receiver hpol LNA power is ON")

        if not ant.sensor.rsc_rxl_lna_v_power_enabled.get_value():
            user_logger.warning("L-band receiver vpol LNA power is not enabled. switch on the vpol LNA power")
        else:
            print(":) receiver vpol LNA power is ON")

    elif ant.sensors.rsc_rxl_rfe1_temperature.get_value() > 30 and ant.sensors.rsc_rxl_rfe1_temperature.get_value() < 100: 
        user_logger.warning("L-band rfe1 temperature is warm @ %.2f. alert the site technician" % (ant.sensors.rsc_rxl_rfe1_temperature.get_value()))
    else:
    	user_logger.warning("L-band rfe1 temperature is warm. alert the site technician")
    
    indexer_angle =  ant.sensor.ap_indexer_position_raw.get_value() 
    print("receiver indexer value is %d." % indexer_angle)
    ril=ant.sensor.ap_indexer_position.get_value().upper()
    if ril not in ["u", "l", "s", "x"]:
        user_logger.warning("AP indexer in unknown position")
    else:
        print("receiver indexer is at the %s-band position" % ril) 

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="AR1 antenna quick check")
parser.add_option("--ant", type=str, default=None,
                  help="Antenna to check in the format m0xx. (default='%default')")

# assume basic options passed from instruction_set
parser.set_defaults(description = "AR1 AP Quick Check")
(opts, args) = parser.parse_args()
if opts.ant is None: 
    raise SystemExit("antenna name required %s" % parser.print_usage())

with verify_and_connect(opts) as kat:
    print("Antenna quick check : start")
    
    ant_active = [ant for ant in kat.ants]
    for ant in ant_active:
        if ant.name == opts.ant:
            any_errors=False
            # pass these first
            if not ant.sensor.ap_connected.get_value():
                raise RuntimeError("AP is not connected")
            if not ant.sensor.comms_ok.get_value():
                raise RuntimeError("AP comms is not okay")
            if ant.sensor.ap_control.get_value() != "remote":
                raise RuntimeError("AP is in %s mode. It cannot be remotely controlled" % ant.sensor.ap_control.get_value())
            if ant.sensor.ap_e_stop_reason.get_value() != "none":
                raise RuntimeError("AP e_stop: %s" % ant.sensor.ap_e_stop_reason.get_value())

            # Verify that all the config data has been added from RTS
            print("\nBeginning a quick check of antenna %s" % ant.name)
            print("\nAP version: %s" % ant.sensor.ap_api_version.get_value())
            if ant.sensor.rsc_rxu_serial_number.get_value():
                print("UHF band serial number: %s" % ant.sensor.rsc_rxu_serial_number.get_value())  
            if ant.sensor.rsc_rxl_serial_number.get_value(): 
                print("L-band serial number: %s" % ant.sensor.rsc_rxl_serial_number.get_value())    
            if ant.sensor.rsc_rxs_serial_number.get_value():
                print("S-band serial number: %s" % ant.sensor.rsc_rxs_serial_number.get_value())     
            if ant.sensor.rsc_rxx_serial_number.get_value():
                print("X-band serial number: %s" % ant.sensor.rsc_rxx_serial_number.get_value()) 
            
            print("\nChecking Receiver")
            check_receivers()
	
            print("\nChecking Digitisers")
            check_digitisers()

            print("\nchecking acu encoder")
            enc=[
                        "ap_azim_enc_failed",
                        "ap_elev_enc_failed"
                ]
            if check_sensors(ant.sensor, enc):
                any_errors = True
            else:
                print(":) ACU encoders OK")

            print("\nchecking antenna key lock switch")
            safe_pos=[
                        "ap_key_switch_safe1_enabled",
                        "ap_key_switch_safe2_enabled"
                     ]
            if check_sensors(ant.sensor, safe_pos):
                any_errors = True
            else:
                print(":) antenna is unlocked")

            print("\nChecking motion")
            amps=[
                        "ap_azim_amp1_failed",
                        "ap_azim_amp2_failed",
                        "ap_azim_motion_error",
                        "ap_azim_servo_failed",
                        "ap_elev_amp_failed",
                        "ap_elev_motion_error",
                        "ap_elev_servo_failed",
                        "ap_ridx_amp_failed"
                 ]
            if check_sensors(ant.sensor, amps):
                any_errors = True
            else:
                print(":) AP Servo OK")

            print("\nChecking brakes")
            breaks=[
                        "ap_azim_brake1_failed",
                        "ap_azim_brake2_failed",
                        "ap_elev_brake_failed",
                        "ap_ridx_brake_failed"
                   ]
            if check_sensors(ant.sensor, breaks):
                any_errors = True
            else:
                print(":) Brakes OK")

            breaks=[
                        "ap_azim_brakes_released",
                        "ap_elev_brakes_released"
                       ]
            if check_sensors(ant.sensor, breaks, test_false=True):
                any_errors = True
            else:
                print(":) Brakes released")
                
            print("\nChecking doors")
            doors=[
                        "ap_hatch_door_open", 
                        "ap_ped_door_open", 
                        "ap_yoke_door_open"
                      ]
            if check_sensors(ant.sensor, doors):
                any_errors = True
            else:
                print(":) All Doors Closed")

            print("\nChecking AP drive limits")
            azim_limits=[
                        "ap_azim_emergency2_limit_ccw_reached",
                        "ap_azim_emergency2_limit_cw_reached",
                        "ap_azim_emergency_limit_ccw_reached",
                        "ap_azim_emergency_limit_cw_reached",
                        "ap_azim_hard_limit_ccw_reached",
                        "ap_azim_hard_limit_cw_reached",
                        "ap_azim_prelimit_ccw_reached",
                        "ap_azim_prelimit_cw_reached",
                        "ap_azim_soft_limit_ccw_reached",
                        "ap_azim_soft_limit_cw_reached",
                        "ap_azim_soft_prelimit_ccw_reached",
                        "ap_azim_soft_prelimit_cw_reached"
                        ]
            if check_sensors(ant.sensor, azim_limits):
                any_errors = True
            else:
                print(":) AP not in Azimuth Limit")

            elev_limits=[
                        "ap_elev_emergency2_limit_down_reached",
                        "ap_elev_emergency2_limit_up_reached",
                        "ap_elev_emergency_limit_down_reached",
                        "ap_elev_emergency_limit_up_reached",
                        "ap_elev_hard_limit_down_reached",
                        "ap_elev_hard_limit_up_reached",
                        "ap_elev_soft_limit_down_reached",
                        "ap_elev_soft_limit_up_reached",
                        "ap_elev_soft_prelimit_down_reached",
                        "ap_elev_soft_prelimit_up_reached"
                        ]
            if check_sensors(ant.sensor, elev_limits):
                any_errors = True
            else:
                print(":) AP not in Elevation Limit")


            if any_errors:
                print('\n') 
		user_logger.warning("Some errors detected, investigate before accepting %s" % ant.name)
            else:
                print("\n{} has passed the handover test".format(opts.ant))
    # -fin-
    print("check_ant_AR1.py: stop\n")
