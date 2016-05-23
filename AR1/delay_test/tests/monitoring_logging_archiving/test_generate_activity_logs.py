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
from tests import settings, fixtures, Aqf, AqfTestCase
from tests import specifics
from tests import utils
from datetime import datetime

from katmisc.utils.sensor_parsing import (
    resources_from_allocations)

@system('all')
class TestActivityLogs(AqfTestCase):

    """Tests windstow."""

    def setUp(self):
        fixtures.sim = self.sim
        fixtures.cam = self.cam
        specifics.set_wind_speeds(self, 9)

    def tearDown(self):
        # Remove what tests setUp has done
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

    @aqf_vr('VR.CM.AUTO.L.56', 'VR.CM.AUTO.L.57')
    def test_activity_log_for_wind_stow(self):
        """Test that an activity log is created for wind stow."""

        speed = 10
        Aqf.step("Ensure ANC mean wind speed is less than %dm/s (%.2fkm/h) "
                 "before starting the test" % (speed, speed * 3.6))
        Aqf.sensor('cam.sensors.anc_mean_wind_speed').wait_until_lt(speed)

        self._point_antennas(120, 45)

        
        trigger_utctime = datetime.utcnow()
        #KAT-7 gust wind triggers at 15.2 and MKAT triggers at 16.9
        new_speed = 17
        Aqf.step("Simulate a gust by setting a gust wind speed of %dm/s "
                 "(%.2fkm/h) on the simulators" % (new_speed, new_speed * 3.6))
        specifics.simulate_wind_gust(self, new_speed+1)

        alarm_name = "ANC_Wind_Gust"
        expected_priority = "critical"
        Aqf.step("Verify that ANC Wind Gust alarm has been raised")
        utils.check_alarm_severity(self, "ANC_Wind_Gust", "critical")

        #2014-11-16 13:41:37.726Z activity INFO alarm_event(controller.py:347) Alarm event ANC_Wind_Gust set ...
        grep_for = ['Alarm event', alarm_name+' set']
        found, utctime = utils.check_activity_logged(self, grep_for, aftertime=trigger_utctime, lines=5000)
        Aqf.step("Checking activity log: %s at %s - after %s" % (found, utctime, trigger_utctime))
        Aqf.is_true(found, "Activity log was generated for windstow event")

        #2014-11-16 13:41:28.77Z alarms ERROR _alert_log(alarm.py:283) new critical alarm - ANC_Wind_Gust (...)
        found, utctime = utils.check_alarm_logged(self, alarm_name, expected_priority, aftertime=trigger_utctime, lines=5000)
        Aqf.step("Checking alarms log: %s at %s - after %s" % (found, utctime, trigger_utctime))
        Aqf.is_true(found, "Alarms log was generated for %s at priority %s" % (alarm_name, expected_priority))

        Aqf.hop("Reset wind gust")
        specifics.reset_wind_gust(self, 9)
        speed = 10

        Aqf.step("Verify that ANC Wind Gust alarm returns to nominal")
        Aqf.sensor('cam.sensors.kataware_alarm_ANC_Wind_Gust').wait_until_status(is_in=['nominal'])

        Aqf.hop("Clear the nominal alarm")
        self.cam.kataware.req.alarm_clear("ANC_Wind_Gust")

        Aqf.step("Wait for windstow release - system interlock to return to NONE")
        Aqf.sensor('cam.sensors.sys_interlock_state').wait_until('NONE', sleep=2, counter=90)

        Aqf.end()

    @system('kat7', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.L.57')
    def test_cam_generates_activity_log_for_alarms(self):
        """Test that the CAM generates activity log for alarms."""

        Aqf.step("Verify that the System_Fire alarm is not active before the test starts")
        if not utils.check_alarm_severity(self, "System_Fire", "nominal"):
            Aqf.skipped("System_Fire alarm is not nominal. Cannot start test.")
            return

        alarm_name = "System_Fire"
        expected_priority = "critical"

        trigger_utctime = datetime.utcnow()
        specifics.simulate_fire(self)
        Aqf.step("Check that the fire alarm has been raised")
        utils.check_alarm_severity(self, alarm_name, expected_priority)

        found, utctime = utils.check_alarm_logged(self, alarm_name, expected_priority, aftertime=trigger_utctime, lines=5000)
        Aqf.step("Checking alarms log: %s at %s - after %s" % (found, utctime, trigger_utctime))
        Aqf.is_true(found, "Alarms log was generated for %s at priority %s" % (alarm_name, expected_priority))
        
        Aqf.hop("Reset the System_Fire alarm")
        specifics.reset_fire(self)

        Aqf.step("Verify that System_Fire alarm returns to nominal")
        Aqf.sensor('cam.sensors.kataware_alarm_System_Fire').wait_until_status(is_in=['nominal'])

        Aqf.step("Clear the System_Fire alarm")
        self.cam.kataware.req.alarm_clear("System_Fire")
        Aqf.end()

    @aqf_vr("VR.CM.AUTO.L.10")
    def test_activity_logs_for_sheduling(self):
        """Test that the CAM generates activity log scheduling."""

        def teardown_subarray():
            utils.teardown_subarray(self, 1)

        subnr = 1
        controlled = specifics.get_controlled_data_proxy(self)
        selected_ants_set = utils.setup_subarray(self, subnr, controlled)

        if not selected_ants_set:
            Aqf.failed("Resource assignment to subarray_1 failed. Aborting test")
            teardown_subarray()
            Aqf.end()
            return

        selected_ant = list(selected_ants_set)[0] # Choose one of the ants
        Aqf.step("Select resources for the test - {} and {}"
                 .format(selected_ant, controlled))
        assignments = self.cam.katpool.sensor.pool_resources_1.get_value()
        if selected_ant in assignments and controlled in assignments:
            Aqf.step("Selected resources are assigned - {} and {}"
                     .format(selected_ant, controlled))
        else:
            Aqf.failed("Selected resources NOT assigned - {} and {} - assignments {}. Aborting test"
                       .format(selected_ants_set, controlled, assignments))
            teardown_subarray()
            Aqf.end()
            return

        if self.cam.sched.sensor.mode_1.get_value() == 'locked':
            Aqf.failed("Scheduler mode is 'locked'. Aborting test")
            teardown_subarray()
            Aqf.end()
            return

        try: #Try the test
            # Create an OBSERVATION schedule block
            Aqf.step("Create two OBSERVATION schedule blocks")
            sb_id_code_1 = utils.create_basic_obs_sb(self, selected_ant, controlled, owner="VR.CM.AUTO.L.10")

            Aqf.step("Set scheduler %s to Manual mode" % (subnr))
            self.cam.sched.req.mode(subnr,"manual")
            Aqf.step("Schedule OBSERVATION SBs %s" % (sb_id_code_1))
            self.cam.sched.req.sb_schedule(subnr,sb_id_code_1)
            count = 60
            scheduled = False
            while (count > 0 and not scheduled):
                count -= 1
                observation_schedule = self.cam.sched.sensor.observation_schedule_1.get_value()
                scheduled = sb_id_code_1 in observation_schedule
                Aqf.wait(2, "Waiting for SBs %s to verify and be scheduled" % (sb_id_code_1))
            if not scheduled:
                Aqf.failed("OBSERVATION SBs %s was not scheduled. Exiting test." % (sb_id_code_1))
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SBs %s were scheduled." % (sb_id_code_1))

            trigger_utctime = datetime.utcnow()
            Aqf.step("Execute OBSERVATION SB %s" % sb_id_code_1)
            self.cam.sched.req.sb_execute(1,sb_id_code_1)
            count = 60
            active = False
            while (count > 0 and not active):
                count -= 1
                active_schedule = self.cam.sched.sensor.active_schedule_1.get_value()
                active = sb_id_code_1 in active_schedule
                Aqf.wait(2, "Waiting for SB %s to verify and start executing" % sb_id_code_1)
            if not active:
                Aqf.failed("OBSERVATION SB %s was not activated. Exiting "
                           "test." % sb_id_code_1)
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SB %s was activated." % sb_id_code_1)

            #2014-11-16 14:01:23.165Z activity INFO verify(katscheduler_types.py:1101) SubSched1, SB 20141116-0013: Verifying SB
            #2014-11-16 14:01:30.135Z activity INFO sb_execute(katscheduler_types.py:801) SubSched1, SB 20141116-0013: Activating SB
            #2014-11-16 14:01:39.508Z activity INFO _stop_task(katscheduler_types.py:969) SubSched1, SB 20141116-0013: Stopping SB
            grep_for = ['SB %s' % sb_id_code_1, 'Verifying SB']
            found, utctime = utils.check_activity_logged(self, grep_for, aftertime=trigger_utctime, lines=5000)
            Aqf.step("Checking activity log: %s at %s - after %s" % (found, utctime, trigger_utctime))
            Aqf.is_true(found, "Activity log was generated for verifying SB %s" % sb_id_code_1)
            grep_for = ['SB %s' % sb_id_code_1, 'Activating SB']
            found, utctime = utils.check_activity_logged(self, grep_for, aftertime=trigger_utctime, lines=5000)
            Aqf.step( "Checking activity log: %s at %s - after %s" % (found, utctime, trigger_utctime))
            Aqf.is_true(found, "Activity log was generated for activating SB %s" % sb_id_code_1)

            # Allowing observation to run for few seconds before we graceful stop observations
            Aqf.hop("Waiting for SB %s to settle before stopping" % sb_id_code_1)
            Aqf.wait(15)
            stop_utctime = datetime.utcnow()
            Aqf.step("Inject request to Complete the SB")
            self.cam.sched.req.sb_complete(1, sb_id_code_1)
            Aqf.wait(5, "Wait for input to be processed")

            grep_for = ['SB %s' % sb_id_code_1, 'Stopping SB']
            found, utctime = utils.check_activity_logged(self, grep_for, aftertime=stop_utctime, lines=5000)
            Aqf.step("Checking activity log: %s at %s - after %s" % (found, utctime, stop_utctime))
            Aqf.is_true(found, "Activity log was generated for stopping SB %s" % sb_id_code_1)

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))

        finally:
            teardown_subarray()

        Aqf.end()


