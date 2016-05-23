###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""
from tests import fixtures, Aqf

def simulate_intrusion(aqfbase):
    Aqf.step("Set the ASC intrusion sensor to be not nominal")
    aqfbase.sim.asc.req.set_sensor_value("intrusion.ok",0)

def reset_intrusion(aqfbase):
    Aqf.step("Set the ASC intrusion sensor back to nominal")
    aqfbase.sim.asc.req.set_sensor_value("intrusion.ok", 1)

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
    Aqf.step("Set the ASC fire sensor to an error value")
    aqfbase.sim.asc.req.set_sensor_value("fire.ok",0)

def reset_fire(aqfbase):
    Aqf.step("Set the ASC fire sensor back to nominal")
    aqfbase.sim.asc.req.set_sensor_value("fire.ok", 1)

def simulate_cooling_failure(aqfbase):
    Aqf.step("Set the ASC chiller sensor to an error value")
    aqfbase.sim.asc.req.simulate_value("chiller.water.temperature",35,0,0)
    Aqf.sensor("sim.asc.sensor.chiller_water_temperature").wait_until_status(is_not_in=['nominal'], counter=5)

def reset_cooling_failure(aqfbase):
    Aqf.step("Set the ASC chiller sensor back to nominal")
    aqfbase.sim.asc.req.simulate_value("chiller.water.temperature",15,0,0)
    Aqf.sensor("sim.asc.sensor.chiller_water_temperature").wait_until_status(is_in=['nominal'], counter=5)

def simulate_wind_gust(aqfbase, windspeed):
    Aqf.step("Set the wind_speed on ASC simulator to %s" % (str(windspeed)))
    aqfbase.sim.asc.req.simulate_value("wind.speed", windspeed, 0, 0)

def reset_wind_gust(aqfbase, windspeed):
    Aqf.step("Set the wind_speed on ASC simulator to %s" % (str(windspeed)))
    aqfbase.sim.asc.req.simulate_value("wind.speed", windspeed, 1.0, 2.0)

def set_wind_speeds(aqfbase, windspeed):
    """Set the exact wind speed"""
    Aqf.step("Set the wind_speed on ASC simulator to %s" % (str(windspeed)))
    aqfbase.sim.asc.req.simulate_value("wind.speed", windspeed, 0, 0)
    Aqf.step("Set the wind_speed_2 on ASCcombo simulator to %s" % (str(windspeed)))
    aqfbase.sim.asccombo.req.simulate_value("wind.speed_2", windspeed, 0, 0)

def reset_wind_speeds(aqfbase, windspeed):
    """Reset wind speed with fluctuation"""
    Aqf.step("Reset the wind_speed on ASC simulator to %s" % (str(windspeed)))
    aqfbase.sim.asc.req.simulate_value("wind.speed", windspeed, 1.0, 2.0)
    Aqf.step("Reset the wind_speed_2 on ASCcombo simulator to %s" % (str(windspeed)))
    aqfbase.sim.asccombo.req.simulate_value("wind.speed_2",
                                         windspeed, 1.0, 2.0)
