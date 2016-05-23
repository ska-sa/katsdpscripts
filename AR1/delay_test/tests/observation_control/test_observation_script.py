###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time

from tests import fixtures, settings, specifics, wait_sensor_includes
from tests import wait_sensor_excludes, Aqf, AqfTestCase
from nosekatreport import (aqf_vr, system, slow)

from katcorelib.katobslib.common import ScheduleBlockPriorities, ScheduleBlockTypes
from katcorelib.katobslib.common import ScheduleBlockStates, ScheduleBlockOutcomes
from tests import utils


##############################################################################
### TBD - this file should be reworked for Aqf and for all systems
### as it tests core design concepts of the scheduling and resource allocation
##############################################################################

def check_if_req_ok(aqfbase, obj, message=None):
    """Helper function that check if request was successfully sent"""

    if message:
        Aqf.hop(message)

    msg = obj.messages[0]
    if msg.arguments[0] == 'ok':
        Aqf.progress("Verify that request '%s' is successfully sent"
            % msg.name )
    else:
        Aqf.progress("Failed to send request '%s'" % msg.name)
    ok = Aqf.equals(
        msg.arguments[0], 'ok',
        "Checking request '%s'" % msg.name)
    return ok

@system('all')
class TestObservationScript(AqfTestCase):

    """Test observation script - execution, augmentation and archiving."""

    def setUp(self):
        self.sub_nr = 1
        self.sub = self.cam.subarray_1
        self.sub.req.free_subarray(timeout=20)
        self.controlled_data = specifics.get_controlled_data_proxy(self)
        self.selected_ant = self.cam.ants[-1].name

    def tearDown(self):
        self.sub.req.free_subarray(timeout=20)
        self.cam.katpool.req.free_subarray(self.sub_nr)

    @aqf_vr('CAM_OBSERVE_basic_script')
    def test_basic_script(self):
        """Test the execution of a basic script - moving ants only."""

        # Get other active schedule blocks
        active_sbs = self.cam.sched.sensor.active_schedule_1.get_value()
        sb_id_code = self.obs.sb.new(owner="AQF-test_basic_script", antenna_spec=self.selected_ant,
                                     controlled_resources=self.controlled_data)
        self.obs.sb.description = "Basic script for %s" % self.selected_ant
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        self.obs.sb.instruction_set = "run-obs-script ~/svn/katscripts/cam/basic-script.py -t 5.0 -m 60 --program-block-id=CAM_AQF"
        Aqf.passed('Basic script schedule block created %s.' % sb_id_code)
        time.sleep(2)

        # Free and setup subarray
        Aqf.step("Free subarray %s to start with" % self.sub_nr)
        check_if_req_ok(self, self.sub.req.free_subarray(),
                        "Verify subarray {} state is 'inactive'".format(self.sub_nr))
        Aqf.sensor(self.cam.subarray_1.sensor.state).wait_until("inactive", sleep=1, counter=5)
        resource_csv = ",".join([self.selected_ant,self.controlled_data])
        Aqf.step("Assign resources %s to subarray %s" % (resource_csv, self.sub_nr))
        check_if_req_ok(self, self.sub.req.assign_resources(resource_csv))

        # Activate subarray and set manual scheduling
        Aqf.step("Activate the subarray.")
        check_if_req_ok(self, self.sub.req.activate_subarray(timeout=15),
                "Verify subarray {} state is 'active'".format(self.sub_nr))
        Aqf.sensor(self.cam.sensors.subarray_1_state).wait_until("active", sleep=1, counter=5)
        check_if_req_ok(self, self.sub.req.set_scheduler_mode("manual"),
                "Verify subarray {} scheduler mode is 'manual'".format(self.sub_nr))

        # Assign and schedule SB
        check_if_req_ok(self, self.sub.req.assign_schedule_block(sb_id_code))
        check_if_req_ok(self, self.sub.req.sb_schedule(sb_id_code))

        # Check verification passed and into observation_schedule
        Aqf.step("Wait for SB {} to land in observation schedule.".format(sb_id_code))
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.observation_schedule_1, sb_id_code, 300)

        # Check SB is ready after waiting for SCHED to process
        time.sleep(3)
        self.obs.db_manager.expire()
        sb = self.obs.db_manager.get_schedule_block(sb_id_code)
        self.assertEqual(sb.ready, True)
        Aqf.passed('Schedule block is READY %s.' % sb_id_code)

        # Execute SB
        check_if_req_ok(self, self.sub.req.sb_execute(sb_id_code), "Schedule the SB {}".format(sb_id_code))
        # Check through verification to active
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.active_schedule_1, "%s" % (sb_id_code), 300)
        self.assertTrue(ok, "Schedule block is in active schedule %s" % (sb_id_code))
        # Wait for executor to start the script execution
        time.sleep(2)
        ok = wait_sensor_includes(self.cam, self.cam.exe.sensor.task_states, "%s:%s" % (sb_id_code, "RUNNING"), 10)
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.active_schedule_1, "%s" % (sb_id_code), 3)

        # Check script sensors
        script = 'basic_script.py'
        ok = wait_sensor_includes(self.cam, self.cam.subarray_1.sensor.script_name, 'basic_script.py', 2)
        ok = wait_sensor_includes(self.cam, self.cam.subarray_1.sensor.script_status, 'busy', 2)

        # Wait for completion of this schedule block
        ok = wait_sensor_excludes(self.cam, self.cam.sched.sensor.active_schedule_1, "%s" % (sb_id_code), 600)
        ok = wait_sensor_includes(self.cam, self.cam.exe.sensor.task_states, "%s" % (sb_id_code), 10)
        Aqf.passed('Schedule block has stopped running %s.' % sb_id_code)

        # Check completion
        self.obs.db_manager.expire()
        sb = self.obs.db_manager.get_schedule_block(sb_id_code)
        ok = Aqf.equals(sb.state, ScheduleBlockStates.COMPLETED, 'Check schedule block %s has COMPLETED.' % sb_id_code)
        ok = Aqf.equals(sb.outcome, ScheduleBlockOutcomes.SUCCESS, 'Check schedule block %s outcome is SUCCESS.' % sb_id_code)

        Aqf.hop("Test cleanup - Free subarray")
        self.sub.req.free_subarray()
        
        Aqf.end('Basic script executed successfully!')

    @system("kat7", all=False)
    @aqf_vr('CAM_OBSERVE_kat7_script')
    def test_kat7_script(self):
        """Test the execution of an RFI scan for KAT7."""

        # Get other active schedule blocks
        full_controlled = ",".join([self.controlled_data,"rfe7"])
        active_sbs = self.cam.sched.sensor.active_schedule_1.get_value()
        sb_id_code = self.obs.sb.new(owner="AQF-test_basic_script", antenna_spec=self.selected_ant,
                                     controlled_resources=full_controlled)
        self.obs.sb.description = "RFI scan for %s" % self.selected_ant
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        self.obs.sb.instruction_set = "run-obs-script ~/scripts/observation/rfi_scan.py --stow-when-done -m 120 --proposal-id=OBS_SCRIPT --program-block-id=CAM_AQF"
        Aqf.passed('RFI scan schedule block created %s.' % sb_id_code)
        time.sleep(2)

        # Free and setup subarray
        Aqf.step("Free subarray %s to start with" % self.sub_nr)
        check_if_req_ok(self, self.sub.req.free_subarray(),
                        "Verify subarray {} state is 'inactive'".format(self.sub_nr))
        Aqf.sensor(self.cam.subarray_1.sensor.state).wait_until("inactive", sleep=1, counter=5)
        resource_csv = ",".join([self.selected_ant,full_controlled])
        Aqf.step("Assign resources %s to subarray %s" % (resource_csv, self.sub_nr))
        check_if_req_ok(self, self.sub.req.assign_resources(resource_csv))

        # Activate subarray and set manual scheduling
        Aqf.step("Activate the subarray.")
        check_if_req_ok(self, self.sub.req.activate_subarray(timeout=15),
                "Verify subarray {} state is 'active'".format(self.sub_nr))
        Aqf.sensor(self.cam.sensors.subarray_1_state).wait_until("active", sleep=1, counter=5)
        check_if_req_ok(self, self.sub.req.set_scheduler_mode("manual"),
                "Verify subarray {} scheduler mode is 'manual'".format(self.sub_nr))

        # Assign and schedule SB
        check_if_req_ok(self, self.sub.req.assign_schedule_block(sb_id_code))
        check_if_req_ok(self, self.sub.req.sb_schedule(sb_id_code))

        # Check verification passed and into observation_schedule
        Aqf.step("Wait for SB {} to land in observation schedule.".format(sb_id_code))
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.observation_schedule_1, sb_id_code, 300)

        # Check SB is ready after waiting for SCHED to process
        time.sleep(3)
        self.obs.db_manager.expire()
        sb = self.obs.db_manager.get_schedule_block(sb_id_code)
        self.assertEqual(sb.ready, True)
        Aqf.passed('Schedule block is READY %s.' % sb_id_code)

        # Execute SB
        check_if_req_ok(self, self.sub.req.sb_execute(sb_id_code), "Schedule the SB {}".format(sb_id_code))
        # Check through verification to active
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.active_schedule_1, "%s" % (sb_id_code), 300)
        self.assertTrue(ok, "Schedule block is in active schedule %s" % (sb_id_code))
        # Wait for executor to start the script execution
        time.sleep(2)
        ok = wait_sensor_includes(self.cam, self.cam.exe.sensor.task_states, "%s:%s" % (sb_id_code, "RUNNING"), 10)
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.active_schedule_1, "%s" % (sb_id_code), 3)

        # Check script sensors
        script = 'rfi_scan.py'
        ok = wait_sensor_includes(self.cam, self.cam.subarray_1.sensor.script_name, 'rfi_scan.py', 2)
        ok = wait_sensor_includes(self.cam, self.cam.subarray_1.sensor.script_status, 'busy', 2)

        # Wait for completion of this schedule block
        ok = wait_sensor_excludes(self.cam, self.cam.sched.sensor.active_schedule_1, "%s" % (sb_id_code), 600)
        ok = wait_sensor_includes(self.cam, self.cam.exe.sensor.task_states, "%s" % (sb_id_code), 10)
        Aqf.passed('Schedule block has stopped running %s.' % sb_id_code)

        # Check completion
        self.obs.db_manager.expire()
        sb = self.obs.db_manager.get_schedule_block(sb_id_code)
        ok = Aqf.equals(sb.state, ScheduleBlockStates.COMPLETED, 'Check schedule block %s has COMPLETED.' % sb_id_code)
        ok = Aqf.equals(sb.outcome, ScheduleBlockOutcomes.SUCCESS, 'Check schedule block %s outcome is SUCCESS.' % sb_id_code)

        Aqf.hop("Test cleanup - Free subarray")
        self.sub.req.free_subarray()
        
        Aqf.end('KAT7 RFI scan executed successfully!')

