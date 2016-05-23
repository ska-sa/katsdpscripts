
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

from tests import settings, fixtures, utils, specifics, Aqf, AqfTestCase
from datetime import datetime
from nosekatreport import system, aqf_vr

@system('all')
class TestCriticalAlarms(AqfTestCase):

    """Tests alarms."""

    def setUp(self):
        fixtures.sim = self.sim
        fixtures.cam = self.cam

    def tearDown(self):
        Aqf.hop("Reset the System_Fire trigger")
        specifics.reset_fire(self)
        Aqf.hop("Reset the cooling failure trigger")
        specifics.reset_cooling_failure(self)
        #Aqf.hop("Reset the imminent power failure trigger")
        #specifics.reset_imminent_power_failure(self)
        pass
    @system('kat7', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.I.7', 'VR.CM.AUTO.L.34')
    def test_cam_generates_critical_alarm_for_fire(self):
        """Test that the CAM system generates critical safety alarms for fire.
        """

        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        try:
            Aqf.step("Verify that the System_Fire alarm is not active before the test starts")
            if not utils.check_alarm_severity(self, "System_Fire", "nominal"):
                Aqf.skipped("System_Fire alarm is not nominal. Cannot start test.")
                return
            # set the cc fire.ok sensor to error so we run the test
            run_time = datetime.utcnow()
            specifics.simulate_fire(self)
            Aqf.wait(3, "Wait for alarm trigger to be processed")

            Aqf.hop("Verify that the fire aggregate is now in error")
            Aqf.sensor(self.cam.sensors.agg_system_fire_ok).wait_until(False, sleep=3)

            Aqf.step("Check that the fire alarm has been raised")
            utils.check_alarm_severity(self, "System_Fire", "critical")

            Aqf.is_true(utils.check_alarm_logged(self, "System_Fire", "critical", run_time),
                        "Check that the System_Fire alarm has been logged")

            Aqf.hop("Reset the System_Fire alarm")
            specifics.reset_fire(self)
            Aqf.wait(3, "Wait for alarm reset to be processed")

            utils.check_alarm_severity(self, "System_Fire", "nominal")
            Aqf.step("Clear the System_Fire alarm")
            self.cam.kataware.req.alarm_clear("System_Fire")
        finally:
            Aqf.hop("Reset the System_Fire alarm")
            specifics.reset_fire(self)

      
    @system('kat7', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.I.4', 'VR.CM.AUTO.I.7', 'VR.CM.AUTO.L.34')
    def test_cam_generates_critical_alarm_for_cooling_failure(self):
        """Test that the CAM system generates critical safety alarms for cooling failure.
        """

        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        try:
            Aqf.hop("Verify that the System_Cooling_Failure alarm is not active before the test starts")
            if not utils.check_alarm_severity(self, "System_Cooling_Failure", "nominal"):
                Aqf.skipped("System_Cooling_Failure alarm is not nominal. Cannot start test.")
                return

            run_time = datetime.utcnow()
            specifics.simulate_cooling_failure(self)
            Aqf.wait(3, "Wait for alarm trigger to be processed")

            Aqf.step("Verify that the cooling aggregate is now in error")
            Aqf.sensor(self.cam.sensors.agg_system_cooling_ok).wait_until(False, sleep=3)
           
            Aqf.step("Check that the cooling alarm has been raised")
            utils.check_alarm_severity(self, "System_Cooling_Failure", "critical")

            Aqf.wait(2, "Wait for alarm to be logged")
            Aqf.is_true(utils.check_alarm_logged(self, "System_Cooling_Failure", "critical", run_time),
                        "Check that the System_Cooling_Failure alarm has been logged")

            Aqf.hop("Reset the cooling failure before the shutdown action is triggered")
            specifics.reset_cooling_failure(self)
            Aqf.wait(3, "Wait for alarm reset to be processed")

            utils.check_alarm_severity(self, "System_Cooling_Failure", "nominal")
            Aqf.step("Clear the System_Cooling_Failure alarm")
            self.cam.kataware.req.alarm_clear("System_Cooling_Failure")
        finally:
            Aqf.hop("Reset the cooling failure")
            specifics.reset_cooling_failure(self)

        Aqf.end()

    @system('mkat' , all=False)
    @aqf_vr('VR.CM.AUTO.I.46', 'VR.CM.AUTO.L.34')
    def test_cam_generates_critical_alarm_for_imminent_power_failure(self):
        """Test that the CAM system generates critical safety alarms for imminent power failure.
        """

        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        try:
            Aqf.hop("Verify that the System_Imminent_Power_Failure alarm is not active before the test starts")
            if not utils.check_alarm_severity(self, "System_Imminent_Power_Failure", "nominal"):
                Aqf.skipped("System_Imminent_Power_Failure alarm is not nominal. Cannot start test.")
                return

            run_time = datetime.utcnow()
            specifics.simulate_imminent_power_failure(self)
            Aqf.wait(3, "Wait for alarm trigger to be processed")

            Aqf.step("Verify that the power ok aggregate is now in error")
            Aqf.sensor(self.cam.sensors.agg_system_power_ok).wait_until(False, sleep=3)

            Aqf.step("Check that the imminent power failure alarm has been raised")
            utils.check_alarm_severity(self, "System_Imminent_Power_Failure", "critical")

            Aqf.is_true(utils.check_alarm_logged(self, "System_Imminent_Power_Failure", "critical", run_time),
                        "Check that the System_Imminent_Power_Failure alarm has been logged")

            Aqf.hop("Reset the imminent power failure before the shutdown action is triggered")
            specifics.reset_imminent_power_failure(self)
            Aqf.wait(3, "Wait for alarm reset to be processed")

            utils.check_alarm_severity(self, "System_Imminent_Power_Failure", "nominal")
            Aqf.step("Clear the System_Imminent_Power_Failure alarm")
            self.cam.kataware.req.alarm_clear("System_Imminent_Power_Failure")
        finally:
            Aqf.hop("Reset the imminent power failure before the shutdown action is triggered")
            specifics.reset_imminent_power_failure(self)

        Aqf.end()

    @system('mkat', all=False)
    @aqf_vr('VR.CM.AUTO.I.7', 'VR.CM.AUTO.L.34')
    def test_cam_generates_critical_alarm_for_kapb_temperature(self):
        """Test that the CAM system generates alarm for KAPB temperature out of range.
        """

        # Wait for alarms processing for 2 min
        ok = Aqf.sensor("cam.kataware.sensor.alarms_processing_active").wait_until(True, sleep=1, counter=240)
        if not ok:
            Aqf.failed("Alarms processing not active - aborting test")
            Aqf.exit()

        try:
            Aqf.hop("Verify that the System_KAPB_Temperature_Failure alarm is not active before the test starts")
            if not utils.check_alarm_severity(self, "System_KAPB_Temperature_Failure", "nominal"):
                Aqf.skipped("System_KAPB_Temperature_Failure alarm is not nominal. Cannot start test.")
                return

            run_time = datetime.utcnow()
            specifics.simulate_cooling_failure(self)
            Aqf.wait(3, "Wait for alarm trigger to be processed")

            Aqf.step("Verify that the KAPB temperature ok aggregate is now in error")
            Aqf.sensor(self.cam.sensors.agg_anc_kapb_temperature_ok).wait_until(False, sleep=3)

            Aqf.wait(3, "Wait for alarm trigger to be processed")
            Aqf.step("Check that the System_KAPB_Temperature_Failure alarm has been raised")
            utils.check_alarm_severity(self, "System_KAPB_Temperature_Failure", "critical")

            Aqf.is_true(utils.check_alarm_logged(self, "System_KAPB_Temperature_Failure", "critical", run_time),
                        "Check that the System_KAPB_Temperature_Failure alarm has been logged")

            Aqf.hop("Reset the cooling failure before the shutdown action is triggered")
            specifics.reset_cooling_failure(self)

            Aqf.wait(10, "Wait for alarm reset to be processed (10s update period))")
            utils.check_alarm_severity(self, "System_KAPB_Temperature_Failure", "nominal")
            Aqf.step("Clear the System_KAPB_Temperature_Failure alarm")
            self.cam.kataware.req.alarm_clear("System_KAPB_Temperature_Failure")
        finally:
            Aqf.hop("Reset the cooling failure")
            specifics.reset_cooling_failure(self)

        Aqf.end()
