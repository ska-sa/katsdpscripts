###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""
import random
import time
import tornado

from tornado import gen
from datetime import date
from tornado.websocket import websocket_connect
from katportal.test.jsonrpc_helper import JSONRPCRequest, JSONRPCResponse
from tests import fixtures, wait_sensor_includes, Aqf
from katcorelib.katobslib.common import (ScheduleBlockStates,
                                         ScheduleBlockTypes,
                                         ScheduleBlockPriorities,
                                         ScheduleBlockVerificationStates)

def reset_intrusion(aqfbase):
    Aqf.step("Setting the KAPB intrusion sensor back to nominal.")
    aqfbase.sim.bms.req.set_sensor_value("kapb-rfi-door-open",0)

def simulate_intrusion(aqfbase):
    Aqf.step("Setting the KAPB intrusion sensor to be not nominal.")
    aqfbase.sim.bms.req.set_sensor_value("kapb-rfi-door-open", 1)

def get_controlled_data_proxy(aqfbase, sub_nr=1):
    #For now default to use subarray 1
    return "data_{}".format(sub_nr)

def get_specific_controlled(aqfbase, sub_nr=1, controlled="data"):
    # Return the specific data_N proxy from controlled - Use data_N for subarray_N for mkat always
    if controlled == "data" or controlled == "data_{}".format(sub_nr):
        return "data_{}".format(sub_nr)
    else:
        return ""

def simulate_fire(aqfbase):
    Aqf.step("Setting the BMS KAPB fire sensor to an error value")
    aqfbase.sim.bms.req.set_sensor_value("kapb-fire-active", 1)

def reset_fire(aqfbase):
    Aqf.step("Setting the BMS KAPB fire sensor back to nominal")
    aqfbase.sim.bms.req.set_sensor_value("kapb-fire-active", 0)

def simulate_cooling_failure(aqfbase):
    Aqf.step("Setting the BMS KAPB cooling sensor to an error value")
    aqfbase.sim.bms.req.set_sensor_value("imminent-cooling-failure1", 1)
    aqfbase.sim.bms.req.simulate_value("kapb-temperature1", 35, 0, 0)
    Aqf.sensor("sim.bms.sensor.kapb_temperature1").wait_until_status(is_not_in=['nominal'], counter=5)

def reset_cooling_failure(aqfbase):
    Aqf.step("Setting the BMS KAPB cooling sensor back to nominal")
    aqfbase.sim.bms.req.set_sensor_value("imminent-cooling-failure1", 0)
    aqfbase.sim.bms.req.simulate_value("kapb-temperature1", 22, 0.5, 0.5)
    Aqf.sensor("sim.bms.sensor.kapb_temperature1").wait_until_status(is_in=['nominal'], counter=5)

def simulate_imminent_power_failure(aqfbase):
    Aqf.step("Setting the BMS KAPB imminent power failure sensor to an error value")
    aqfbase.sim.bms.req.set_sensor_value("imminent-power-failure", 1)

def reset_imminent_power_failure(aqfbase):
    Aqf.step("Setting the BMS KAPB imminent power failure sensor back to nominal")
    aqfbase.sim.bms.req.set_sensor_value("imminent-power-failure", 0)
    
def simulate_wind_gust(aqfbase, windspeed):
    Aqf.step("Setting the wind-speed on WIND simulator to %s" % (str(windspeed)))
    aqfbase.sim.wind.req.simulate_value("wind-speed", windspeed, 0, 0)

def reset_wind_gust(aqfbase, windspeed):
    Aqf.step("Setting the wind-speed on WIND simulator to %s" % (str(windspeed)))
    aqfbase.sim.wind.req.simulate_value("wind-speed", windspeed, 1.0, 2.0)

def set_wind_speeds(aqfbase, windspeed):
    """Set the exact wind speed"""
    Aqf.step("Setting the wind-speed on WIND simulator to %s" % (str(windspeed)))
    aqfbase.sim.wind.req.simulate_value("wind-speed", windspeed, 0, 0)
    Aqf.step("Setting the wind-speed on WEATHER simulator to %s" % (str(windspeed)))
    aqfbase.sim.weather.req.simulate_value("wind-speed", windspeed, 0, 0)

def reset_wind_speeds(aqfbase, windspeed):
    """Reset wind speed with fluctuation"""
    Aqf.step("Resetting the wind-speed on WIND simulator to %s" % (str(windspeed)))
    aqfbase.sim.wind.req.simulate_value("wind-speed", windspeed, 1.0, 2.0)
    Aqf.step("Resetting the wind-speed on WEATHER simulator to %s" % (str(windspeed)))
    aqfbase.sim.weather.req.simulate_value("wind-speed", windspeed, 1.0, 2.0)

