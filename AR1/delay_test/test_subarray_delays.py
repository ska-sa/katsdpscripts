#!/usr/bin/env python

###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time
import os

from tests import specifics, wait_sensor_includes
from tests import wait_sensor_excludes, Aqf, AqfTestCase
from nosekatreport import (aqf_vr, system)

from katcorelib.katobslib.common import ScheduleBlockTypes
from katcorelib.katobslib.common import ScheduleBlockStates, ScheduleBlockOutcomes

def check_if_req_ok(aqfbase, obj, message=None):
    """Helper function that check if request was successfully sent"""

    if message:
        Aqf.hop(message)

    msg = obj.messages[0]
    if msg.arguments[0] == 'ok':
        Aqf.progress("Verify that request '%s' is successfully sent"
                     % msg.name)
    else:
        Aqf.progress("Failed to send request '%s'" % msg.name)
    ok = Aqf.equals(
        msg.arguments[0], 'ok',
        "Checking request '%s'" % msg.name)
    return ok

@system('mkat')
class SubarrayScript(AqfTestCase):

    """Test observation script - execution, augmentation and archiving."""
    def setUp(self):
        self._use_sim = False
        try:
            self.selected_ant = os.environ['TESTING_ANT']
            ant_names = ([ant.name for ant in self.cam.ants])
	    ant_names.append('available')
            if self.selected_ant not in ant_names:
                raise KeyError
        except KeyError:
            Aqf.step('No ants specified - using {}.'.format(self.cam.ants[-1].name))
            self.selected_ant = self.cam.ants[-1].name

        try:
            self.sub_nr = int(os.environ['TESTING_SUB_NR'])
        except KeyError:
            Aqf.step('No sub_nr specified - using subarray_1.')
            self.sub_nr = 1

        try:
            self.band = os.environ['TESTING_BAND']
        except KeyError:
            Aqf.step('No band specified - using l-band.')
            self.band = 'l'

        try:
            self.product = os.environ['TESTING_PRODUCT']
        except KeyError:
            Aqf.step('No product specified - using c856M4k.')
            self.product = 'c856M4k'

        try:
            self.instruction_set = os.environ['TESTING_INSTRUCTION_SET']
        except KeyError:
            Aqf.step('No instruction set specified - using "run-obs-script ~/svn/katscripts/cam/basic_capture_start.py"')
            self.instruction_set = "run-obs-script ~/svn/katscripts/cam/basic_capture_start.py"

        try:
            self.instruction_set_args = os.environ['TESTING_ARGS']
        except KeyError:
            self.instruction_set_args = ""

        try:
            self.description = os.environ['TESTING_DESCRIPTION']
        except KeyError:
            self.description = "Basic script"

        self.controlled_data = None
        try:
            self.use_data = int(os.environ['TESTING_USE_DATA'])
            if self.use_data:
                self.controlled_data = specifics.get_controlled_data_proxy(self, self.sub_nr)
        except KeyError:
            self.controlled_data = specifics.get_controlled_data_proxy(self, self.sub_nr)

        self.sub = getattr(self.cam, 'subarray_{}'.format(self.sub_nr))
#         self.sub.req.free_subarray(timeout=20)

    def tearDown(self):
        self.sub.req.free_subarray(timeout=20)
        self.cam.katpool.req.free_subarray(self.sub_nr)

    @aqf_vr('CAM_OBSERVE_basic_script')
    def test_basic_script(self):
        """Test the execution of a basic script - moving ants only."""
        # Get other active schedule blocks
        sb_id_code = self.obs.sb.new(owner="AQF-test_delay_script",
                                     antenna_spec=self.selected_ant,
                                     controlled_resources=self.controlled_data)
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        self.obs.sb.instruction_set = "{} {}".format(self.instruction_set, self.instruction_set_args)
        self.obs.sb.description = self.description

        Aqf.passed('Basic script schedule block created %s.' % sb_id_code)

#         # Free and setup subarray
#         Aqf.step("Free subarray %s to start with" % self.sub_nr)
#         check_if_req_ok(self, self.sub.req.free_subarray(),
#                         "Verify subarray {} state is 'inactive'".format(self.sub_nr))
#         Aqf.sensor(self.sub.sensor.state).wait_until("inactive", sleep=1, counter=5)
#         if self.controlled_data:
#             resource_csv = ",".join([self.selected_ant, self.controlled_data])
#         else:
#             resource_csv = self.selected_ant
#         Aqf.step("Assign resources %s to subarray %s" % (resource_csv, self.sub_nr))
#         check_if_req_ok(self, self.sub.req.assign_resources(resource_csv))
#
#         check_if_req_ok(self, self.sub.req.set_band(self.band),
#                         "Verify subarray {} band is '{}'".format(self.sub_nr, self.band))
#         check_if_req_ok(self, self.sub.req.set_product(self.product),
#                         "Verify subarray {} product is '{}'".format(self.sub_nr, self.product))
#         # Activate subarray and set manual scheduling
#         Aqf.step("Activate the subarray.")
#         check_if_req_ok(self, self.sub.req.activate_subarray(timeout=60),
#                         "Verify subarray {} state is 'active'".format(self.sub_nr))
#         Aqf.sensor(self.sub.sensor.state).wait_until("active", sleep=1, counter=5)
#         check_if_req_ok(self, self.sub.req.set_scheduler_mode("manual"),
#                         "Verify subarray {} scheduler mode is 'manual'".format(self.sub_nr))

        # Assign and schedule SB
        check_if_req_ok(self, self.sub.req.assign_schedule_block(sb_id_code))
        check_if_req_ok(self, self.sub.req.sb_schedule(sb_id_code))

        # Check verification passed and into observation_schedule
        Aqf.step("Wait for SB {} to land in observation schedule.".format(sb_id_code))
        observation_schedule = getattr(self.cam.sched.sensor,
                                       'observation_schedule_{}'.format(self.sub_nr))
        ok = wait_sensor_includes(self.cam, observation_schedule, sb_id_code, 300)

        # Check SB is ready after waiting for SCHED to process
        time.sleep(3)
        self.obs.db_manager.expire()
        sb = self.obs.db_manager.get_schedule_block(sb_id_code)
        self.assertEqual(sb.ready, True)
        Aqf.passed('Schedule block is READY %s.' % sb_id_code)

        # Execute SB
        check_if_req_ok(
            self, self.sub.req.sb_execute(sb_id_code),
            "Schedule the SB {}".format(sb_id_code))
        # Check through verification to active
        ok = wait_sensor_includes(
            self.cam,
            getattr(self.cam.sched.sensor, 'active_schedule_{}'.format(self.sub_nr)),
            "%s" % (sb_id_code), 300)
        self.assertTrue(ok, "Schedule block is in active schedule %s" % (sb_id_code))
        # Wait for executor to start the script execution
        time.sleep(2)
        ok = wait_sensor_includes(self.cam, self.cam.exe.sensor.task_states, "%s:%s" % (sb_id_code, "RUNNING"), 10)
        ok = wait_sensor_includes(
            self.cam,
            getattr(self.cam.sched.sensor, 'active_schedule_{}'.format(self.sub_nr)),
            "%s" % (sb_id_code), 3)

        # Check script sensors
        ok = wait_sensor_includes(self.cam, self.sub.sensor.script_name, 'basic_script.py', 2)
        ok = wait_sensor_includes(self.cam, self.sub.sensor.script_status, 'busy', 2)

        # Wait for completion of this schedule block
        ok = wait_sensor_excludes(
            self.cam,
            getattr(self.cam.sched.sensor, 'active_schedule_{}'.format(self.sub_nr)),
            "%s" % (sb_id_code), 600)
        ok = wait_sensor_includes(self.cam, self.cam.exe.sensor.task_states, "%s" % (sb_id_code), 10)
        Aqf.passed('Schedule block has stopped running %s.' % sb_id_code)

        # Check completion
        self.obs.db_manager.expire()
        sb = self.obs.db_manager.get_schedule_block(sb_id_code)
        ok = Aqf.equals(sb.state, ScheduleBlockStates.COMPLETED, 'Check schedule block %s has COMPLETED.' % sb_id_code)
        ok = Aqf.equals(sb.outcome, ScheduleBlockOutcomes.SUCCESS, 'Check schedule block %s outcome is SUCCESS.' % sb_id_code)

        Aqf.hop("Test cleanup - Free subarray")
        self.sub.req.free_subarray(timeout=60)

        Aqf.end('Basic script executed successfully!')
 # -fin-
