###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""

import traceback

from nosekatreport import system, aqf_vr, slow
from tests import settings, fixtures, utils, specifics, Aqf, AqfTestCase

@system('all')
class TestWindstow(AqfTestCase):

    """Tests windstow."""

    def setUp(self):
        fixtures.sim = self.sim
        fixtures.cam = self.cam
        specifics.set_wind_speeds(self, 9)

    def tearDown(self):
        pass

    def _point_antennas(self, az, el):
        """Point antennas to given az and el."""
        Aqf.step("Point the antennas to make them move")
        az = float(az)
        el = float(el)
        for ant in self.cam.ants:
            ant.req.target_azel(az, el)
            ant.req.mode("POINT")

        Aqf.hop("Verify that antennas are pointing")
        for ant in self.cam.ants:
            Aqf.sensor(ant.sensor.mode).wait_until('POINT', sleep=1)
            Aqf.equals('POINT', ant.sensor.mode.get_value(),
                       'Check that antenna %s is pointing' % ant.name)

    def _check_antennas_ready(self):
        for ant in self.cam.ants:
            mode = ant.sensor.mode.get_value()
            Aqf.equals("STOP", mode, 'Antenna %s not stowed' % ant.name)
            Aqf.sensor(ant.sensor.windstow_active).eq(False)

    def _check_one_antenna_ready(self, ant_name, timeout=60):
        # Cycle every 2 seconds
        sleep = 2
        counter = timeout/2
        Aqf.sensor("cam.%s.sensor.mode" % ant_name).wait_until("STOP", sleep=sleep, counter=counter)
        
    @slow
    @aqf_vr('VR.CM.AUTO.I.3', 'VR.CM.AUTO.I.6')
    def test_automatic_stop_for_mean_wind(self):
        """Test that antennas is stowed on high mean wind speed.

        Test that an alarm is raised, antennas stowed and observations stopped
        for mean wind speed conditions.

        """

        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        subnr = 1
        speed = 10
        Aqf.step("Ensure ANC mean wind speed is less than %dm/s (%.2fkm/h) "
                 "before starting the test" % (speed, speed * 3.6))
        Aqf.sensor('cam.sensors.anc_mean_wind_speed').wait_until_lt(speed)

        self._point_antennas(123, 80)

        #Choose a speed higher than mean wind speed and lower than gust speed (m/s)
        if settings.system == "kat7":
            new_speed = 15
            wait_speed = 12.1  #KAT-7 mean wind triggers at 12.0
        elif settings.system == "mkat_rts":
            new_speed = 16
            wait_speed = 11.2  #MeerKAT mean wind triggers at 11.1
        elif settings.system == "mkat":
            new_speed = 16
            wait_speed = 11.2  #MeerKAT mean wind triggers at 11.1

        Aqf.step("Force the mean wind speed higher by setting wind speed "
                 "sensors on simulators to high value: %dm/s (%.2fkm/h)" %
                 (new_speed, new_speed * 3.6))
        specifics.set_wind_speeds(self, new_speed)

        Aqf.step("Wait for ANC mean wind speed to reach %dm/s (%.2fkm/h)" %
                 (wait_speed, wait_speed * 3.6))
        Aqf.sensor('cam.sensors.anc_mean_wind_speed').wait_until_gt(wait_speed, counter=200)

        Aqf.wait(3, "Wait for alarm trigger to be processed")
        Aqf.step("Verify that ANC mean windspeed alarm has been raised")
        utils.check_alarm_severity(self, "ANC_Wind_Speed", "critical")

        Aqf.step("Verify that antennas are stowed")
        for ant in self.cam.ants:
            Aqf.sensor(ant.sensor.mode).wait_until('STOW', sleep=1)
            Aqf.sensor(ant.sensor.windstow_active).wait_until(True, sleep=1)

        Aqf.step("Verify that Scheduler is still ready (not locked)")
        ready = utils.check_scheduler_ready(self, subnr)
        if ready:
            Aqf.passed("Scheduler %s is ready - not mode 'locked'" % subnr)
        else:
            Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % subnr)
        
        specifics.reset_wind_speeds(self, 6)

        speed = 10
        Aqf.step("Wait for ANC mean wind speed to return to normal")
        Aqf.sensor('cam.sensors.anc_mean_wind_speed').wait_until_lt(speed, counter=200)

        Aqf.step("Verify that ANC mean windspeed alarm returns to nominal")
        Aqf.sensor('cam.sensors.kataware_alarm_ANC_Wind_Speed').wait_until_status(is_in=['nominal'])

        Aqf.progress("Wait for UNSTOW timeout (2min) to elapse")
        self._check_one_antenna_ready(self.cam.ants[0].name, timeout=120)

        Aqf.step("Verify that antennas are ready to continue "
                "(STOP and not windstow_active)")
        self._check_antennas_ready()

        Aqf.step("Verify that Scheduler is ready (not locked)")
        ready = utils.check_scheduler_ready(self, subnr)
        if ready:
            Aqf.passed("Scheduler %s is ready - not mode 'locked'" % subnr)
        else:
            Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % subnr)

        Aqf.hop("Clear the nominal alarm")
        self.cam.kataware.req.alarm_clear("ANC_Wind_Speed")
        specifics.reset_wind_speeds(self, 9)
        Aqf.end()

    @aqf_vr('VR.CM.AUTO.I.3', 'VR.CM.AUTO.I.6')
    def test_automatic_stop_for_gust_wind(self):
        """Test antenna stow on high wind gust.

        Test that an alarm is raised, antennas stowed and observations stopped
        for gust wind speed conditions.

        """

        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        subnr = 1
        speed = 10
        Aqf.step("Ensure ANC mean wind speed is less than %dm/s (%.2fkm/h) "
                 "before starting the test" % (speed, speed * 3.6))
        Aqf.sensor('cam.sensors.anc_mean_wind_speed').wait_until_lt(speed)

        self._point_antennas(120, 45)

        #KAT-7 gust wind triggers at 15.2 and MKAT triggers at 16.9
        new_speed = 17
        Aqf.step("Simulate a gust by setting a gust wind speed of %dm/s "
                 "(%.2fkm/h) on the simulators" % (new_speed, new_speed * 3.6))
        specifics.simulate_wind_gust(self, new_speed+1)

        Aqf.progress("Waiting for the ANC gust wind speed sensor to reach "
                     "%dm/s (%.2fkm/h) " % (new_speed, new_speed * 3.6))
        Aqf.sensor('cam.sensors.anc_gust_wind_speed'
                   ).wait_until_gt(new_speed, 0.5, counter=200)

        Aqf.step("Verify that ANC Wind Gust alarm has been raised")
        utils.check_alarm_severity(self, "ANC_Wind_Gust", "critical")

        Aqf.step("Verify that antennas are stowed")
        for ant in self.cam.ants:
            Aqf.sensor(ant.sensor.mode).wait_until('STOW', sleep=1)

        Aqf.step("Verify that Scheduler is ready (not locked)")
        ready = utils.check_scheduler_ready(self, subnr)
        if ready:
            Aqf.passed("Scheduler %s is ready - not mode 'locked'" % subnr)
        else:
            Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % subnr)

        specifics.reset_wind_gust(self, 9)
        speed = 10
        Aqf.step("Wait for ANC gust wind speed to return to normal")
        Aqf.sensor('cam.sensors.anc_gust_wind_speed').wait_until_lt(speed, counter=200)

        Aqf.step("Verify that ANC Wind Gust alarm returns to nominal")
        Aqf.sensor('cam.sensors.kataware_alarm_ANC_Wind_Gust').wait_until_status(is_in=['nominal'], counter=60)

        Aqf.progress("Wait for UNSTOW timeout (2min) to elapse")
        self._check_one_antenna_ready(self.cam.ants[0].name, timeout=150)

        Aqf.step("Verify that antennas are ready to continue "
                "(STOP and not windstow_active)")
        self._check_antennas_ready()

        Aqf.step("Verify that Scheduler is ready (not locked)")
        ready = utils.check_scheduler_ready(self, subnr)
        if ready:
            Aqf.passed("Scheduler %s is ready - not mode 'locked'" % subnr)
        else:
            Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % subnr)

        Aqf.hop("Clear the nominal alarm")
        self.cam.kataware.req.alarm_clear("ANC_Wind_Gust")
        Aqf.end()

