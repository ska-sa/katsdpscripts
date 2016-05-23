###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""

import time

from tests import settings, utils, Aqf, AqfTestCase
from nosekatreport import system, aqf_vr


@system('all')
class TestAlarms(AqfTestCase):

    """Tests alarms."""

    def setUp(self):
        if settings.system in ("kat7"):
            self.sim.asc.req.set_sensor_value('ups.battery.not.low', 1)
            time.sleep(6)
            for ant in self.cam.ants:
                ant.req.inhibit(0)
            self.cam.kataware.req.alarm_clear('ASC_UPS_Battery_Low')
            self.cam.kataware.req.alarm_clear('ant2_chiller_motor')
            self.cam.kataware.req.alarm_clear("ASC_air_temp")
        if settings.system in ("mkat_rts"):
            self.cam.kataware.req.alarm_clear("ASC_air_temp")

        Aqf.log_status('-' * 30)
        Aqf.log_status("cam.anc.synced={0}".format(self.cam.anc.synced))
        Aqf.log_status('-' * 30)


    def tearDown(self):
        for ant in self.cam.ants:
            ant.req.inhibit(0)
        if settings.system in ("kat7"):
            self.sim.asc.req.set_sensor_value('ups.battery.not.low', 1)
            time.sleep(6)
            self.cam.kataware.req.alarm_clear('ASC_UPS_Battery_Low')
            self.cam.kataware.req.alarm_clear('ant2_chiller_motor')
        if settings.system in ("kat7", "mkat_rts"):
            self.cam.kataware.req.alarm_clear("ASC_air_temp")

    @system('kat7', all=False)
    @aqf_vr('CAM_ALARMS_based_on_aggregate')
    def test_alarm_on_agg_sensor(self):
        """Check handling of an alarm based on an agg sensor."""
        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        # Force the sensor and alarm to be OK  and cleared at the start
        self.sim.bms2.req.set_sensor_value('chiller.motor.not.tripped', 1)
        self.cam.sensors.ant2_bms_chiller_motor_not_tripped.set_strategy('event')
        self.cam.sensors.agg_ant2_chiller_not_tripped.set_strategy('event')
        Aqf.equals(self.cam.sensors.ant2_bms_chiller_motor_not_tripped.status, 'nominal', '')
        Aqf.equals(self.cam.sensors.agg_ant2_chiller_not_tripped.status, 'nominal', '')
        Aqf.equals(self.cam.sensors.agg_ant2_chiller_not_tripped.value, True, '')
        self.cam.kataware.req.alarm_clear('ant2_chiller_motor')
        Aqf.wait(3)
        Aqf.equals(
            self.cam.kataware.sensor.alarm_ant2_chiller_motor.get_value(),
                'nominal,cleared,agg_ant2_chiller_not_tripped '
                "value = True. status = nominal.", '')

        self.sim.bms2.req.set_sensor_value('chiller.motor.not.tripped', 0)
        Aqf.wait(1)
        Aqf.equals(self.cam.sensors.agg_ant2_chiller_not_tripped.value, False, '')
        # Wait because kataware uses period 5 strategy on the sensor.
        Aqf.wait(6)
        Aqf.equals(self.cam.kataware.sensor.alarm_ant2_chiller_motor.get_value(),
                'critical,new,agg_ant2_chiller_not_tripped '
                'is False', '')

        self.cam.kataware.req.alarm_ack('ant2_chiller_motor')
        Aqf.wait(0.5)
        Aqf.equals(
            self.cam.kataware.sensor.alarm_ant2_chiller_motor.get_value(),
                'critical,acknowledged,agg_ant2_chiller_not_tripped '
                'is False', '')

        self.sim.bms2.req.set_sensor_value('chiller.motor.not.tripped', 1)
        Aqf.wait(1)
        Aqf.equals(self.cam.sensors.ant2_bms_chiller_motor_not_tripped.status,
                   'nominal', '')
        Aqf.equals(self.cam.sensors.agg_ant2_chiller_not_tripped.value,
                   True, '')
        self.cam.kataware.req.alarm_clear('ant2_chiller_motor')
        Aqf.wait(2)
        alarm = self.cam.kataware.sensor.alarm_ant2_chiller_motor
        Aqf.equals(alarm.get_value(),
                   'nominal,cleared,agg_ant2_chiller_not_'
                   'tripped value = True. status = nominal.', '')

        #TBD - Use katlogserver to check alarms.log (grep-log or tail-log)
        Aqf.end()

    @system('kat7', all=False)
    @aqf_vr('CAM_ALARMS_based_on_float_sensor')
    def test_alarm_on_float_sensor(self):
        """Test handling of an alarm based on a float sensor."""

        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        #Note - this test fails as the asc simulator
        # changes the values of the air temp too quickly
        Aqf.step("Verify that asc air temperature is nominal to start with")
        Aqf.sensor("sim.sensors.asc_asc_air_temperature").set(15.0)
        Aqf.sensor("sim.sensors.asc_asc_air_temperature"
                   ).wait_until_status(is_in=['nominal'], counter=5)
        utils.check_alarm_severity(self, "ASC_air_temp", "nominal")
        Aqf.step("Simulate an asc air temperature in warning")
        Aqf.sensor("sim.sensors.asc_asc_air_temperature").set(33.0)
        # Wait for kataware to recognise the alarm
        Aqf.wait(6, "Waiting for kataware to process the alarm")
        utils.check_alarm_severity(self, "ASC_air_temp", "warn")
        Aqf.wait(10)
        Aqf.step("Simulate an asc air temperature in error")
        Aqf.sensor("sim.sensors.asc_asc_air_temperature").set(36.0)
        # Wait for kataware to recognise the alarm
        Aqf.wait(6, "Waiting for kataware to process the alarm")
        utils.check_alarm_severity(self, "ASC_air_temp", "error")
        Aqf.wait(10)
        Aqf.step("Simulate an asc air temperature in critical")
        Aqf.sensor("sim.sensors.asc_asc_air_temperature").set(39.0)
        # Wait for kataware to recognise the alarm
        Aqf.wait(5)
        utils.check_alarm_severity(self, "ASC_air_temp", "critical")
        Aqf.wait(10)

        Aqf.step("Put asc air temperature back to nominal")
        Aqf.sensor("sim.sensors.asc_asc_air_temperature").set(18.0, 1.0, 2.0)
        # Wait for kataware to recognise the alarm
        Aqf.wait(5)
        utils.check_alarm_severity(self, "ASC_air_temp", "nominal")

        Aqf.step("Clear the nominal alarm")
        self.cam.kataware.req.alarm_clear("ASC_air_temp")

        Aqf.end()
