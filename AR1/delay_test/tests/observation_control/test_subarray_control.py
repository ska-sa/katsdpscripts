###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import json
import os
import time
import traceback
import logging
import random

from datetime import datetime

from katconf.sysconf import KatObsConfig

from katmisc.utils import csv_sets
from katcorelib.katobslib.test import test_manager
from katcorelib.katobslib.common import (ScheduleBlockStates,
                                         ScheduleBlockTypes,
                                         ScheduleBlockPriorities,
                                         ScheduleBlockVerificationStates)

from nosekatreport import (aqf_vr, system)

from tests import (wait_sensor, wait_sensor_includes, wait_sensor_excludes, Aqf,
                   AqfTestCase, specifics, utils)

logger = logging.getLogger(__name__)


@system('all')
class TestSubarrayControl(AqfTestCase):

    """Tests for MeerKAT observation control - with subarrays and schedule
    blocks."""

    def set_up_db(self, params_list):

        for params in params_list:
            resource_spec = self.db_manager.create_resource_spec(
                params.get('ants', 'available'),
                params.get('controlled', self.controlled))
            schedule_block = self.db_manager.create_schedule_block(
                resource_spec=resource_spec.id,
                owner=params.get('owner', None),
                sb_type=params.get('type', ScheduleBlockTypes.OBSERVATION),
                priority=params.get('priority', ScheduleBlockPriorities.LOW),
                desired_start_time=params.get('desired_start_time', None),
                expected_duration_seconds=params.get('expected_duration_seconds', None),
                instruction_set=params.get('instruction_set', None))
            self.schedule_block_id_codes.append(schedule_block.id_code)

    def setUp(self):
        # Clear all subarrays and resources
        # Set the controlled resource
        self.sub_nr =  1
        self.controlled =  specifics.get_controlled_data_proxy(self.sub_nr)
        # Select an antenna and device to use
        ants = self.get_component_from_cam('ants')
        self.ant_proxy = self.cam.ants[0].name
        self.extra_proxy = self.cam.ants[1].name
        if len(self.cam.ants) > 2:
            self.other_extra_proxy = self.cam.ants[2].name
        else:
            self.other_extra_proxy = None

        # Handle mkat and kat7 receptors
        self.ant_device = "antenna" if self.ant_proxy.startswith("ant") else "ap"
        self.schedule_block_id_codes = []
        config = KatObsConfig(self.cam.system)
        # Create a katobs database manager for this system configuration.
        self.db_manager = test_manager.KatObsDbTestManager(config.db_uri)

        # Set strategies on some sensors
        # self.cam.sched.set_sampling_strategies(filter="", strat_and_params='event')
        # self.cam.exe.set_sampling_strategies(filter="", strat_and_params='event')
        self.subarrays = [sub for sub in self.cam.katpool.sensor.subarrays.get_value().split(",")]
        self.sub_objs = {}
        for sub_nr in self.subarrays:
            self.sub_objs[int(sub_nr)] = getattr(self.cam, "subarray_%s" % sub_nr)

    def tearDown(self):
        Aqf.step("tearDown - done")

    def check_verification_state(self, id_code, verification_state):
        """Test verification result of schedule block."""
        # Get schedule block from db
        self.db_manager.expire()
        schedule_block = self.db_manager.get_schedule_block(id_code)
        # Check not verified
        ok = Aqf.equals(schedule_block.verification_state, verification_state,
                   'Check sb %s verification state is %s.' %
                   (id_code, verification_state.key))
        return ok

    def check_sb_state(self, id_code, schedule_state):
        """Test scheduling state of a schedule block."""
        # Get schedule block from db
        self.db_manager.expire()
        schedule_block = self.db_manager.get_schedule_block(id_code)
        # Check the state
        ok = Aqf.equals(schedule_block.state, schedule_state,
                   'Check that sb {} state is {}'
                   .format(id_code, schedule_state.key))
        return ok

    def check_task_log(self, id_code, is_dry_run=False):
        """Check that outfile for task has been created.

        Parameters
        ----------
        id_code : str
            Schedule block ID code.
        is_dry_run : bool
            True if for dry-run, False if for normal task execution."""

        ok = False
        obs_node_ip = self.cam.nodes.obs.ip if self.cam.nodes.get('obs') else self.cam.nodes.monctl.ip
	ssh = utils.ssh_server(obs_node_ip)
        now = datetime.utcnow()

        outfile = ("/var/kat/tasklog/%02d/%02d/%02d/%s/%s" %
                   (now.year, now.month, now.day, id_code,
                    "dryrun.out" if is_dry_run else "progress.out"))
	stdin, stdout, stderr = ssh.exec_command("ls -l %s" % outfile, timeout=20)
	stdout = stdout.readlines()
	stderr = stderr.readlines()
	if not stderr and stdout and outfile in stdout[0]:
            Aqf.passed('The progress sb %s has outfile %s' % (id_code, outfile))
            ok = True
        return ok

    def check_progress_log_for_string(self, id_code, string):
        """Check that outfile for task has been created.

        Parameters
        ----------
        id_code : str
            Schedule block ID code.
	string: str
	    String to search progress file for
	"""

        ok = False
        obs_node_ip = self.cam.nodes.obs.ip if self.cam.nodes.get('obs') else self.cam.nodes.monctl.ip
	ssh = utils.ssh_server(obs_node_ip)
        now = datetime.utcnow()

        outfile = ("/var/kat/tasklog/%02d/%02d/%02d/%s/%s" %
                   (now.year, now.month, now.day, id_code, "progress.out"))
	stdin, stdout, stderr = ssh.exec_command("cat %s" % outfile)
	stderr = stderr.readlines()
	stdout = stdout.readlines()
	result = []
	if not stderr:
            result = [l.strip() for l in stdout
                      if l.startswith(string)]
	    
        return result

    def check_verification(self, id_code, success):
        """Test requests verification for a schedule block. Checks result.

        Parameters
        ----------
        id_code : str
            Schedule block ID code.
        success : bool
            Expected success of verification.
        """
        if success:
            verification_state = ScheduleBlockVerificationStates.VERIFIED
            task_state = "COMPLETED"
        else:
            verification_state = ScheduleBlockVerificationStates.FAILED
            task_state = "FAILED"

        # Request verification
        reply = self.cam.sched.req.sb_verify(1, id_code)
        # self.assertEqual(reply.succeeded, success)
        Aqf.equals(reply.succeeded, success, 'Check the request verification')
        Aqf.passed('Schedule block %s verification request succeeded.' % id_code)
        if success:
            # Wait for verification to completed
            ok = wait_sensor_includes(
                self.cam, self.cam.exe.sensor.dryrun_task_states,
                "%s:%s" % (id_code, task_state), 180)
            # self.assertTrue(ok, 'cam.exe dryrun_task_states does not contain %s:%s' % (id_code, task_state))
            Aqf.equals(ok, True,
                       'Check cam.exe dryrun_task_states contains %s:%s' %
                       (id_code, task_state))
            Aqf.passed('Schedule block %s verification %s.' %
                       (id_code, task_state))
        time.sleep(5)
        ok = self.check_verification_state(id_code, verification_state)
        return ok

    def check_verification_declined(self, id_code):
        """Test verification result of schedule block."""
        # Request verification
        reply = self.cam.sched.req.sb_verify(1, id_code)
        # Check verified: ok
        ok = Aqf.equals(reply.succeeded, False,
                   'Verify that sb %s verification is declined.' %
                   id_code)
        return ok

    def schedule_and_check_pass(self, sub_nr,
                                schedule_block_id_code, timeout=30):
        """Test for success of scheduling a schedule block."""
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        schedule_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        schedule_sensor_obj = getattr(self.cam.sched.sensor,
                                      schedule_sensor_name)
        reply = subarray.req.sb_schedule(schedule_block_id_code)
        Aqf.equals(reply.succeeded, True, 'Verify schedule request succeeded '
                   'for schedule block %s.' % schedule_block_id_code)

        self.db_manager.expire()
        schedule_block = self.db_manager.get_schedule_block(
            schedule_block_id_code)
        if schedule_block.type == ScheduleBlockTypes.OBSERVATION:
            # Wait for verify to complete
            ok = wait_sensor_includes(
                self.cam, schedule_sensor_obj,
                schedule_block_id_code, timeout=timeout)
            Aqf.is_true(ok, 'Verify cam.sched.sensor.{schedule_sensor_name} includes '
                        '{schedule_block_id_code}'.format(**locals()))
            self.db_manager.expire()
            schedule_block = self.db_manager.get_schedule_block(
                schedule_block_id_code)
            ok = Aqf.equals(schedule_block.verification_state,
                       ScheduleBlockVerificationStates.VERIFIED,
                       'Verify schedule block %s is VERIFIED.' %
                       schedule_block_id_code)
            if not ok:
                Aqf.failed('SB %s verification FAILED' % schedule_block_id_code)
                return False
        # Check that SB state is SCHEDULED
        self.db_manager.expire()
        schedule_block = self.db_manager.get_schedule_block(
            schedule_block_id_code)
        ok = Aqf.equals(schedule_block.state, ScheduleBlockStates.SCHEDULED,
                   'Verify schedule block %s is SCHEDULED.' %
                   schedule_block_id_code)
        if not ok:
            Aqf.failed('SB %s state not sCHEDULED' % schedule_block_id_code)
            return False
        return ok

    def execute_and_check_pass(self, sub_nr, schedule_block_id_code,
                               type=ScheduleBlockTypes.OBSERVATION, timeout=100):
        """Test for success of executing a schedule block."""
        Aqf.hop('Verify schedule block %s can be executed' %
                schedule_block_id_code)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        result = subarray.req.sb_execute(schedule_block_id_code)
        active_schedule_sensor_name = 'active_schedule_{}'.format(sub_nr)
        # active_schedule_sensor_obj = getattr(self.cam.sched.sensor,
        #                                     'active_schedule_{}'.format(sub_nr))
        if result.succeeded:
            Aqf.passed('Execute request successful schedule block %s' %
                       schedule_block_id_code)
        else:
            Aqf.failed('Shedule block %s could not be executed' %
                       schedule_block_id_code)
            return False

        # Check to see that SB state is ACTIVE
        Aqf.hop('Verify that {0} is ACTIVE; waiting up to {1}s for {0} to show '
                'up in {2} sensor'.
                format(schedule_block_id_code, timeout,
                       active_schedule_sensor_name))
        ok = wait_sensor_includes(self.cam,
                                  'sched_active_schedule_{}'.format(sub_nr),
                                  schedule_block_id_code,
                                  timeout=timeout)
        if not ok:
            Aqf.failed('Shedule block %s not in sched_active_schedule_%s' %
                       (schedule_block_id_code, sub_nr))
            return False

        allocations = []
        allocations_str = self.cam.katpool.sensor.allocations.get_value()
        if not allocations_str == '':
            allocations = [allocation[0] for allocation in
                           json.loads(allocations_str)
                           if allocation[1] == schedule_block_id_code]
        ok = Aqf.is_true(len(allocations) > 0,
                    'Check that allocation list is more than 0')
        if ok:
            Aqf.passed('Schedule block %s has resources allocated to it' %
                   schedule_block_id_code)
        else:
            Aqf.failed('Shedule block %s does not have expected allocated resources' %
                       schedule_block_id_code)
            return False

        if type == ScheduleBlockTypes.OBSERVATION:
            ok = self.check_sb_running(sub_nr, schedule_block_id_code, timeout)
            ok = ok and self.check_task_log(schedule_block_id_code)
            if ok:
                Aqf.passed('Schedule block %s is executing.' %
                       schedule_block_id_code)
            else:
                Aqf.failed('Schedule block %s execution failed.' %
                       schedule_block_id_code)
        return ok

    def check_sb_running(self, sub_nr, schedule_block_id_code, timeout=180):
        sb_running_expected_value = '{}:{}'.format(schedule_block_id_code, 'RUNNING')
        Aqf.hop('Waiting up to {}s for sb {} to start running by checking '
                'sensor exe_task_states.'.
                format(timeout, schedule_block_id_code))
        ok = wait_sensor_includes(self.cam, self.cam.sensors.exe_task_states,
                                  sb_running_expected_value, timeout=timeout)
        # ##    'sb {} in "RUNNING" state'.format(schedule_block_id_code))
        Aqf.is_true(ok, 'Verify sb {} in "RUNNING" state'.
                    format(schedule_block_id_code))

        active_schedule_sensor_name = 'sched_active_schedule_{}'.format(sub_nr)
        Aqf.hop('Verify that {0} is ACTIVE; waiting up to {1}s for {0} to show '
                'up in {2} sensor'.
                format(schedule_block_id_code, timeout,
                       active_schedule_sensor_name))
        ok = wait_sensor_includes(self.cam, active_schedule_sensor_name,
                                  schedule_block_id_code,
                                  timeout=timeout)
        ok = Aqf.is_true(ok, 'Check that {} is in cam.sched.sensor.{}'.format(
            schedule_block_id_code, active_schedule_sensor_name))
        return ok

    def wait_sb_complete(self, sub_nr, schedule_block_id_code,
                         type=ScheduleBlockTypes.OBSERVATION,
                         is_running_check=True, timeout=300):
        """Wait for currently active schedule_block to complete"""
        Aqf.step('Wait for sb {} to complete'.format(schedule_block_id_code))
        active_schedule_sensor_name = 'sched_active_schedule_{}'.format(sub_nr)
        # active_schedule_sensor = getattr(
        #    self.cam.sensors, active_schedule_sensor_name)
        if is_running_check:
            # First check that the sb is actually running
            running = wait_sensor_includes(self.cam,
                                           active_schedule_sensor_name,
                                           schedule_block_id_code,
                                           timeout=timeout)
            Aqf.is_true(running, 'Schedule block {} is running'.
                        format(schedule_block_id_code))

        Aqf.hop('Waiting up to {}s for sb {} to complete'
                .format(timeout, schedule_block_id_code))
        completed = wait_sensor_excludes(self.cam, active_schedule_sensor_name,
                                         schedule_block_id_code,
                                         timeout=timeout)
        ok = Aqf.is_true(completed, 'Verify that sb {} completed within {}s'.format(
            schedule_block_id_code, timeout))

        # Also check that it is no longer running on katexecuter if it is an
        # observation script
        if type == ScheduleBlockTypes.OBSERVATION:
            exe_timeout = 10
            sb_running = '{}:RUNNING'.format(schedule_block_id_code)
            Aqf.hop('Waiting up to {}s for sb {} to be stop running on '
                    'katexecutor'.format(exe_timeout, schedule_block_id_code))
            not_running = wait_sensor_excludes(self.cam, "exe_task_states",
                                               sb_running, timeout=exe_timeout)
            ok = Aqf.is_true(not_running,
                        'Verify observation sb {} no longer executing on Executor'.
                        format(schedule_block_id_code))
        return ok

    def execute_and_check_fail(self, sub_nr, schedule_block_id_code):
        """Test for failure of executing a schedule block."""
        Aqf.step('Verify that schedule block %s cannot be executed' %
                 schedule_block_id_code)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        result = subarray.req.sb_execute(schedule_block_id_code)
        if not result.succeeded:
            Aqf.passed('Not executing schedule block %s' %
                       schedule_block_id_code)
            ok = True
        else:
            Aqf.failed('Schedule block %s should not have been executed' %
                       schedule_block_id_code)
            ok = False
        return ok

    def create_sb_with_user_interface(self, user):
        """Creates a schedule block, using the user library.

        Parameters
        ----------
        user : str
            User name under which the schedule block is created.

        Returns
        -------
        sb_id_code : str
            ID code of new schedule block.
        """
        sb_id_code = self.obs.sb.new(owner=user, antenna_spec=self.ant_proxy)
        return sb_id_code

    def check_if_req_ok(self, obj):
        """Helper function that check if request was successfully sent"""

        msg = obj.messages[0]
        if msg.arguments[0] == 'ok':
            Aqf.progress("Verify if request '%s' is successfully sent" %
                         msg.name)
        else:
            # log to the user
            Aqf.progress("Failed to send request '%s'" % msg.name)
        ok = Aqf.equals(msg.arguments[0], 'ok',
                        "Verify if request '%s' was sent successfully" %
                        msg.name)
        return ok

    def _create_obs_sb(self, selected_ant, controlled,
                       program_block, runtime=180, instruction_set=None,
                       description=None):
        sb_id_code = self.obs.sb.new(owner="aqf-test",
                                     antenna_spec=selected_ant,
                                     controlled_resources=controlled)
        self.obs.sb.description = "Basic Track for %s" % selected_ant if description is None else description
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        if not instruction_set:
            self.obs.sb.instruction_set = (
                "run-obs-script ~/svn/katscripts/cam/basic-script.py -t 3 -m {runtime} "
                "--proposal-id=CAM_AQF --program-block-id={program_block}".
                format(**locals()))
        else:
            self.obs.sb.instruction_set = (
                "{instruction_set} --program-block-id={program_block}".
                format(**locals()))
        return sb_id_code

    def _create_subarray(self, sub_nr, subarr_spec):
        specific_controlled = specifics.get_specific_controlled(subarr_spec["controlled"])
        resource_csv = ",".join(subarr_spec["receptors"] +
                                [specific_controlled])
        # products = subarr_spec["products"]
        # band = subarr_spec["band"]
        self.sub_objs[sub_nr].req.assign_resources(resource_csv)
        # TODO: self.sub_objs[sub_nr].req.set_data_product_configuration()
        # TODO: self.sub_objs[sub_nr].req.set_band()
        return self.sub_objs[sub_nr]

    def setup_default_array(self, sub_nr, activate=True):
        """Set up a default subarray spec for specified subarray number

        Returns the subarray spec dictionary
        """
        # resource_str = ",".join([self.ant_proxy, self.controlled])
        subarr_spec = {"receptors": [self.ant_proxy],  # list of receptor resource names
                       "controlled": 'data',
                       "products": ["32K"],  # CBF data product name (no array name)
                       "band": "l"}
        self._create_subarray(sub_nr, subarr_spec)

        if activate:
            # Activating subarray
            Aqf.step("Activating subarray %s" % (sub_nr))
            Aqf.is_true(self.sub_objs[sub_nr].req.activate_subarray(timeout=100).succeeded,
                        "Activation request for subarray {} successful".
                        format(sub_nr))

        return subarr_spec

    @aqf_vr('VR.CM.AUTO.OBM.5')
    def test_cam_restart_components_for_maintenance(self):
        """VR.CM.AUTO.OBM.5: Test reset and restart components remotely for maintenance
        Description:	Test the CAM system allows the control authority to
        remotely reset or restart equipment during maintenance.
           1. Create a maintenance subarray.
           2. Verify the resetting and restarting of equipment in the maintenance subarray.
        R.CM.FC.22 Reset & restart equipment remotely for maintenance
        CAM shall enable the control authority of a maintenance sub-array to reset or restart equipment in the sub-array remotely.
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        try:
            Aqf.step("Test the CAM system allows the control authority to remotely reset or restart equipment during maintenance.")
            Aqf.step("1. Create a maintenance subarray")
            sub_nr = 2
            Aqf.step("Setup subarray %s to maintenance" % sub_nr)
            self.sub_objs[sub_nr].req.free_subarray(timeout=30)
            self.cam.katpool.req.set_subarray_maintenance(sub_nr, True)
            Aqf.step("Flag receptor %s in-maintenance" % self.ant_proxy)
            self.cam.katpool.req.set_resources_in_maintenance(self.ant_proxy, True)
            Aqf.step("Assign receptor %s to maintenance subarray %s" % (self.ant_proxy, sub_nr))
            self.sub_objs[sub_nr].req.assign_resources(self.ant_proxy)

            Aqf.step("Activate the subarray.")
            self.check_if_req_ok(self.sub_objs[sub_nr].req.activate_subarray(timeout=100))
            Aqf.step("Verify subarray %s state is 'active'" % (sub_nr))
            ok = Aqf.sensor(self.cam.subarray_2.sensor.state).wait_until("active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.step("2. Now restart of equipment in the maintenance subarray. (%s, %s)" % (self.ant_proxy, self.ant_device))
            self.check_if_req_ok(self.cam.subarray_2.req.restart_maintenance_device(self.ant_proxy, self.ant_device))

            Aqf.hop("Test cleanup - Free subarray")
            self.sub_objs[sub_nr].req.free_subarray(timeout=30)
            Aqf.hop("Test cleanup - Take receptor out of maintenance")
            self.cam.katpool.req.set_resources_in_maintenance(self.ant_proxy, False)

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.SC.45')
    def test_cam_capture_info_for_obs_report(self):
        """VR.CM.AUTO.SC.45: Test CAM capture information for SB observation report
        Description:
            Test that CAM captures all the information required for the
            observation report for each scheduling block:

            1. Set up a sub-array.
            2. Create an observation scheduling block.
            3. Schedule and execute the scheduling block.
            4. Extract the scheduling block information from katobs and verify
               that the following has been captured:
                  a. Start and end time of the scheduling block.
                  b. Version of Instrumental Configuration Data that was used.
                  c. Scheduling block status (completed/cancelled/stopped)
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        observation_schedule_name = ("sched_observation_schedule_{}".
                                     format(sub_nr))
        active_schedule_name = "sched_active_schedule_{}".format(sub_nr)
        # subarray_management = getattr(self.cam, 'subarray_{}_management'.format(sub_nr))

        Aqf.step("Using subarray=%s, receptor=%s, controlled=%s" % (
            sub_nr, self.ant_proxy, self.controlled))
        try:  # Wrap test in try-finally so we can ensure proper Aqf finalisation
            Aqf.step("Test that CAM captures all the information required for "
                     "the observation report for each scheduling block: ")
            Aqf.step("1. Set up a sub-array")
            self.setup_default_array(sub_nr, activate=True)
            # Create an OBSERVATION schedule block
            Aqf.step("2. Create an observation scheduling block.")
            sb_id_code_1 = self._create_obs_sb(self.ant_proxy, self.controlled,
                                               "VR.CM.AUTO.SC.45", runtime=10)

            subarray.req.activate_subarray(timeout=100)
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until(
                "active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                raise RuntimeError(msg)

            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set subarray scheduler to manual mode")

            # Assign SB to subarray 1
            Aqf.step("Assign newly created sb %s to subarray %s" %
                     (sb_id_code_1, sub_nr))
            subarray.req.assign_schedule_block(sb_id_code_1)

            Aqf.step("3. Schedule and execute the scheduling block.")

            Aqf.hop('Schedule SB {}'.format(sb_id_code_1))

            subarray.req.sb_schedule(sb_id_code_1)

            schedule_wait_timeout = 60
            Aqf.hop('Waiting up to {}s for SB {} to verify and be scheduled'
                    .format(schedule_wait_timeout, sb_id_code_1))
            scheduled = wait_sensor_includes(self.cam,
                                             observation_schedule_name,
                                             sb_id_code_1,
                                             schedule_wait_timeout)
            if not scheduled:
                msg = ("OBSERVATION SBs (%s) was not scheduled. "
                       "Exiting test." % (sb_id_code_1))
                Aqf.failed(msg)
                raise RuntimeError(msg)

            Aqf.step("Execute OBSERVATION SB %s" % sb_id_code_1)
            subarray.req.sb_execute(sb_id_code_1)
            Aqf.hop("Waiting up to {}s for SB {} to verify and start executing"
                    .format(schedule_wait_timeout, sb_id_code_1))
            active = wait_sensor_includes(self.cam, active_schedule_name,
                                          sb_id_code_1, schedule_wait_timeout)
            if not active:
                msg = ("OBSERVATION SB %s was not activated. Aborting "
                       "test." % sb_id_code_1)
                Aqf.failed(msg)
                raise RuntimeError(msg)
            else:
                Aqf.passed("OBSERVATION SB %s was activated." % sb_id_code_1)

            # Allowing observation to run for few seconds before we gracefully
            # stop observations
            Aqf.step("Allow OBSERVATION SB %s to run for a few seconds "
                     "before stopping the observation" % sb_id_code_1)
            self.check_sb_running(sub_nr, sb_id_code_1)
            Aqf.wait(5)

            Aqf.step("Stop observation SB %s" % sb_id_code_1)
            subarray.req.sb_complete(sb_id_code_1)
            Aqf.hop('Waiting up to {}s for for observation SB {} to complete'
                    .format(60, sb_id_code_1))
            self.wait_sb_complete(sub_nr, sb_id_code_1, timeout=300)

            Aqf.step("4. Extract the scheduling block information from katobs "
                     "and verify that the following has been captured:")
            self.db_manager.expire()  # Force fresh read from db
            self.obs.sb.load(sb_id_code_1, allow_load=True)
            Aqf.step(" a. Start and end time of the scheduling block.")
            Aqf.is_true(self.obs.sb.actual_start_time is not None,
                        'Check that the actual_start_time was saved.')
            Aqf.is_true(self.obs.sb.actual_end_time is not None,
                        'Check that the actual_end_time was saved.')
            Aqf.step(" b. Version of Instrumental Configuration Data "
                     "that was used.")
            Aqf.is_true(self.obs.sb.config_label,
                        'Check that the config_label was saved.')
            Aqf.step(" c. Scheduling block status (completed/cancelled/stopped)")
            Aqf.is_true((self.obs.sb.outcome is not None and
                         self.obs.sb.outcome.key is not 'UNKNOWN'),
                        'Check that the SB outcome was saved.')

        except Exception as err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))

        try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
        finally:
            Aqf.end()

    @aqf_vr('VR.CM.AUTO.OBR.48')
    def test_cam_add_resources_to_subarray(self):
        """
        Using the Lead Operator Interface of a sub-array, verify the following:
        1. Add a resource to the resource pool of any sub-array until the
            sub-array is activated.
        2. Remove a resource from the resource pool of any sub-array until the
            sub-array is activated.
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        try:
            subarray_pool_sensor = getattr(
                self.cam.katpool.sensor, 'pool_resources_{}'.format(sub_nr))
            Aqf.step("Using subarray=%s, receptor=%s, controlled=%s" % (
                sub_nr, self.ant_proxy, self.controlled))

            Aqf.step("Using the Lead Operator Interface of a sub-array, verify "
                     "the following:")
            Aqf.step("1. Add a resource to the resource pool of any sub-array "
                     "until the sub-array is activated.")
            Aqf.step("Set up a sub-array.")
            # resource_str = ",".join([self.ant_proxy, self.controlled])
            subarr_spec = {"receptors": [self.ant_proxy],  # list of receptor resource names
                           "controlled": [self.controlled],
                           "products": ["32K"],  # CBF data product name (no array name)
                           "band": "l"}
            self._create_subarray(sub_nr, subarr_spec)

            # Prepare subarray
            expected_resources = set([self.ant_proxy, self.controlled])
            Aqf.step("Preparing subarray %s" % (sub_nr))
            self._create_subarray(sub_nr, subarr_spec)
            Aqf.equals(csv_sets.set_from_csv(subarray_pool_sensor.get_value()),
                       expected_resources,
                       "Verify that correct resouces are assigned to pool_resources_{}"
                       .format(sub_nr))

            # Assign extra resources
            expected_resources.add(self.extra_proxy)
            Aqf.step("Assign extra resource %s to subarray %s" % (self.extra_proxy, sub_nr))
            reply = subarray.req.assign_resources(self.extra_proxy)
            Aqf.is_true(reply.succeeded,
                        "Verify that resources assignment request was successful")
            Aqf.equals(csv_sets.set_from_csv(subarray_pool_sensor.get_value()),
                       expected_resources,
                       "Verify that correct resouces are assigned to pool_resources_{}"
                       .format(sub_nr))

            # Activating subarray
            Aqf.step("Activating subarray %s" % (sub_nr))
            reply = subarray.req.activate_subarray(timeout=100)
            Aqf.is_true(reply.succeeded, "Verify that subarray activation request succeeded.")

            Aqf.step("Verify that no more resources can be assigned to the subarray")
            # Try assigning extra resources
            Aqf.step("Try assigning resource %s to subarray %s after activated" % (
                self.extra_proxy, sub_nr))

            if self.other_extra_proxy:
                reply = subarray.req.assign_resources(self.other_extra_proxy)
                Aqf.is_true(not reply.succeeded,
                            "Verify that resource assignment request failed.")
                Aqf.equals(csv_sets.set_from_csv(subarray_pool_sensor.get_value()),
                           expected_resources,
                           "Verify that no extra resources were assigned to pool_resources_{}."
                           .format(sub_nr))

            Aqf.step("Free subarray")
            # Freeing subarray
            subarray.req.free_subarray(timeout=30)
            Aqf.is_true(subarray.req.free_subarray(timeout=30).succeeded,
                        "Verify that freeing request succeeded")
            Aqf.equals(subarray.sensor.state.get_value(), 'inactive',
                       "Verify that subarray {} was freed.".format(sub_nr))

            Aqf.step("2. Remove a resource from the resource pool of any sub-array until "
                     "the sub-array is activated.")

            Aqf.step('Recreate subarray {}'.format(sub_nr))
            subarr_spec["receptors"].append(self.extra_proxy)
            expected_resources = set([self.ant_proxy, self.controlled, self.extra_proxy])
            self._create_subarray(sub_nr, subarr_spec)
            Aqf.equals(csv_sets.set_from_csv(subarray_pool_sensor.get_value()),
                       expected_resources,
                       "Verify that correct resouces are assigned to pool_resources_{}."
                       .format(sub_nr))

            Aqf.step("Remove resource from subarray.")
            expected_resources.remove(self.extra_proxy)
            Aqf.step("Unassign resource {} to subarray {}"
                     .format(self.extra_proxy, sub_nr))
            reply = subarray.req.unassign_resources(self.extra_proxy)
            Aqf.is_true(reply.succeeded, "Verify the extra resources was unassigned")
            Aqf.equals(csv_sets.set_from_csv(subarray_pool_sensor.get_value()),
                      expected_resources,
                       "Verify that correct resouces were unassigned from pool_resources_{}"
                       .format(sub_nr))

            # Activating subarray
            Aqf.step("Activating subarray %s" % (sub_nr))
            subarray.req.activate_subarray(timeout=100)

            Aqf.step("Verify that no more resources can be removed from the subarray")

            Aqf.step("Try removing resource {} from subarray {}"
                     .format(self.ant_proxy, sub_nr))
            Aqf.step("Unassign resource {} to subarray {}".format(self.ant_proxy, sub_nr))
            reply = subarray.req.unassign_resources(self.ant_proxy)
            Aqf.is_true(not reply.succeeded, "Verify the resource could not be unassigned")
            Aqf.equals(csv_sets.set_from_csv(subarray_pool_sensor.get_value()),
                       expected_resources,
                       "Verify that no resouces were unassigned from pool_resources_{}"
                       .format(sub_nr))

            # Free the subarray
            Aqf.step("Free subarray %s" % (sub_nr))
            subarray.req.free_subarray(timeout=30)

            # Test cleanup
            Aqf.hop("Reset test environment")
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.OBM.47')
    def test_cam_excludes_maint_components(self):
        """Test CAM excludes maintenance components from future observations"""

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = 1
        sub_nr_2 = 2
        subarray_1 = "subarray_%s" % sub_nr
        subarray_2 = "subarray_%s" % sub_nr_2
        ant_1 = self.ant_proxy
        ant_2 = self.extra_proxy
        Aqf.step("Using subarrays=%s,%s, receptors=%s,%s" % (
            sub_nr, sub_nr_2, ant_1, ant_2))
        sub1 = getattr(self.cam, subarray_1)
        sub2 = getattr(self.cam, subarray_2)

        try: # Try the test

            Aqf.step("Flag {} as maintenance".format(subarray_2))
            self.cam.subarray_2.req.set_subarray_maintenance(True)
            Aqf.step("Flag {} as in-maintenance".format(ant_2))
            self.cam.katpool.req.set_resources_in_maintenance(ant_2, True)
            time.sleep(1) # Wait for signal to process

            maint_subs = self.cam.katpool.sensor.subarrays_maintenance.get_value().split(",")
            maint_resources = self.cam.katpool.sensor.resources_in_maintenance.get_value().split(",")
            Aqf.is_true(str(sub_nr_2) in maint_subs,
                "Verify that katpool reflects {} as maintenance".format(subarray_2))
            Aqf.is_true(ant_2 in maint_resources,
                "Verify that katpool reflects resource {} as in-maintenance".format(ant_2))

            reply = sub1.req.assign_resources(ant_2)
            Aqf.equals(reply.succeeded, False, "Verify that assigning in-maintenance {} to not maintenance {} fails".format(ant_2,subarray_1))
            reply = sub2.req.assign_resources(ant_2)
            Aqf.equals(reply.succeeded, True, "Verify that assigning in-maintenance {} to maintenance {} succeeds".format(ant_2,subarray_2))

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
            raise
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.OBR.49')
    def test_cam_manage_resource_pools(self):
        """
        Verify that resources can be assigned to different subarray resource pools
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = 1
        sub_nr_2 = 2
        if "kat7" not in self.cam.katconfig.site:
            controlled_1 = "data_%s" % sub_nr
            controlled_2 = "data_%s" % sub_nr_2
        else:
            controlled_1 = "dbe7"
            controlled_2 = ""
        Aqf.step("Using subarrays=%s,%s, receptors=%s,%s, controlled=%s,%s" % (
            sub_nr, sub_nr_2, self.ant_proxy, self.extra_proxy,
            controlled_1, controlled_2))

        try: # Try the test

            Aqf.step("Using the Lead Operator Interface of a sub-array, verify the following: ")
            Aqf.step("1. Add a resource to the resource pool of any sub-array until the sub-array is activated. ")
            Aqf.step("Set up subarray %s" % sub_nr)
            subarr_spec = {"receptors": [], # list of receptor resource names
                "controlled": [],
                "products": ["32K"], # CBF data product name (no array name)
                "band": "l"}
            sub1 = self._create_subarray(sub_nr, subarr_spec)
            Aqf.step("Set up subarray %s" % sub_nr_2)
            subarr_spec = {"receptors": [], # list of receptor resource names
                "controlled": [],
                "products": ["32K"], # CBF data product name (no array name)
                "band": "l"}
            sub2 = self._create_subarray(sub_nr_2, subarr_spec)

            # Assign resources to two subarrays
            Aqf.step("Assign resource %s to subarray %s" % (self.ant_proxy, sub_nr))
            reply = sub1.req.assign_resources(self.ant_proxy)
            Aqf.step("Assign resource %s to subarray %s" % (controlled_1 , sub_nr))
            reply = sub1.req.assign_resources(controlled_1)

            Aqf.step("Assign resource %s to subarray %s" % (self.extra_proxy, sub_nr_2))
            reply = sub2.req.assign_resources(self.extra_proxy)
            Aqf.step("Assign resource %s to subarray %s" % (controlled_2, sub_nr_2))
            reply = sub2.req.assign_resources(controlled_2)

            Aqf.hop("Verify that resource pools correcly reflects the assignments")
            pool1_res = self.cam.katpool.sensor.pool_resources_1.get_value().split(",")
            pool2_res = self.cam.katpool.sensor.pool_resources_2.get_value().split(",")
            Aqf.is_true(
                self.ant_proxy in pool1_res and controlled_1 in pool1_res,
                "Verify that resources were correctly assigned to pool-resources-1")
            if controlled_2:
                Aqf.is_true(
                    self.extra_proxy in pool2_res and controlled_2 in pool2_res,
                    "Verify that resources was assigned to pool-resources-2")
            else:
                Aqf.is_true(
                    self.extra_proxy in pool2_res,
                    "Verify that resources was assigned to pool-resources-2")
        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
            raise
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.OBR.49')
    def test_cam_manage_maintenance_resources(self):
        """
        Verify that resources flagged as in-maintenance can be assigned only
        to maintenance subarray
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = 1
        sub_nr_2 = 2
        subarray_1 = "subarray_%s" % sub_nr
        subarray_2 = "subarray_%s" % sub_nr_2
        ant_1 = self.ant_proxy
        ant_2 = self.extra_proxy
        Aqf.step("Using subarrays=%s,%s, receptors=%s,%s" % (
            sub_nr, sub_nr_2, ant_1, ant_2))
        sub1 = getattr(self.cam, subarray_1)
        sub2 = getattr(self.cam, subarray_2)

        try: # Try the test

            Aqf.step("Flag {} as maintenance".format(subarray_2))
            self.cam.subarray_2.req.set_subarray_maintenance(True)
            Aqf.step("Flag {} as in-maintenance".format(ant_2))
            self.cam.katpool.req.set_resources_in_maintenance(ant_2, True)
            time.sleep(1) # Wait for signal to process

            maint_subs = self.cam.katpool.sensor.subarrays_maintenance.get_value().split(",")
            maint_resources = self.cam.katpool.sensor.resources_in_maintenance.get_value().split(",")
            Aqf.is_true(str(sub_nr_2) in maint_subs,
                "Verify that katpool reflects {} as maintenance".format(subarray_2))
            Aqf.is_true(ant_2 in maint_resources,
                "Verify that katpool reflects resource {} as in-maintenance".format(ant_2))

            Aqf.step("Verify that assigning in-maintenance {} to not maintenance {} fails".format(ant_2,subarray_1))
            reply = sub1.req.assign_resources(ant_2)
            Aqf.equals(reply.succeeded, False, 'Verify assign request failed')
            Aqf.step("Verify that assigning in-maintenance {} to maintenance {} succeeds".format(ant_2,subarray_2))
            reply = sub2.req.assign_resources(ant_2)
            Aqf.equals(reply.succeeded, True, 'Verify assign request succeeded')

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
            raise
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.OBR.49')
    def test_cam_excludes_faulty_resources(self):
        """
        Verify the following:
        1. Create a sub-array with at least two receptors.
        2. Put one of the receptors into faulty.
        3. Activate the sub-array.
        4. Perform a point-source scan and verify that the faulty receptor is not included.
        5. On completion of the observation, free the subarray.
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        schedule_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        ###subarray_management = getattr(self.cam, 'subarray_{}_management'.format(sub_nr))
        ###subarray_management.set_sampling_strategies(filter="", strat_and_params='event')
        try:
            Aqf.step("1. Create a sub-array with at least two receptors. ")
            Aqf.hop('Create subarray_{}'.format(sub_nr))
            self.setup_default_array(sub_nr, activate=False)
            Aqf.is_true(subarray.req.assign_resources(self.extra_proxy).succeeded,
                        "Assign 2nd antenna to subarray_{}".format(sub_nr))
            print self.cam.katpool.sensor.pool_resources_1.get_value()
            Aqf.step("2. Put 2nd receptor into faulty. ")
            subarray.req.set_resources_faulty(self.extra_proxy, 1)
            Aqf.step("3. Activate the sub-array. ")
            Aqf.is_true(subarray.req.activate_subarray(timeout=100).succeeded,
                        "Activation request for subarray {} successful".format(sub_nr))
            Aqf.step("4. Execute a basic-script and verify that the faulty receptor "
                     "is not included. ")
            sb_id = self._create_obs_sb(self.ant_proxy, self.controlled,
                                        "VR.CM.AUTO.OBR.49", runtime=10)
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=10)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set it to Manual Scheduling Mode.")
            Aqf.hop("Assign newly created sb %s to subarray %s" % (sb_id, sub_nr))
            subarray.req.assign_schedule_block(sb_id)
            Aqf.hop('Schedule sb {}'.format(sb_id))
            self.schedule_and_check_pass(sub_nr, sb_id)
            Aqf.hop('Execute sb {}'.format(sb_id))
            self.execute_and_check_pass(sub_nr, sb_id)
            self.wait_sb_complete(sub_nr, sb_id, timeout=300)
	    controlled_line = self.check_progress_log_for_string(sb_id, 'Controlled objects: ')
            # now = datetime.utcnow()
            # progress_filename = "/var/kat/tasklog/%02d/%02d/%02d/%s/progress.out" % (
            #     now.year, now.month, now.day, sb_id)
            # with open(progress_filename) as f:
            #     controlled_line = [l.strip() for l in f.readlines()
            #                        if l.startswith('Controlled objects: [')]
            Aqf.step("5. On completion of the observation, free the subarray.")
            # Will be done below in the finally statement
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.SC.50')
    def test_cam_manual_and_queue_scheduling_modes(self):
        """
        Verify that CAM provides Manual and Queue Scheduling Modes for each sub-array:
        1. Create a sub-array.
        2. Set it to Manual Scheduling Mode.
        3. Create and schedule three scheduling blocks.
        4. Verify that the scheduling blocks do not start executing.
        5. Manually execute a scheduling block and wait for completion.
        6. Set the sub-array to Queue Scheduling Mode.
        7. Verify that the ready scheduling blocks start execution through queue
           scheduling and wait for completion.
        8. Tear down the sub-array.
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        schedule_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        try:
            subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
            Aqf.step("Verify that CAM provides Manual and Queue Scheduling Modes for "
                     "each sub-array: ")
            Aqf.step("1. Create a sub-array. ")
            self.setup_default_array(sub_nr, activate=True)
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "2. Set it to Manual Scheduling Mode.")

            Aqf.step("3. Create and schedule three scheduling blocks. ")
            Aqf.hop("Creating three new schedule blocks ")
            sb_id_codes = []
            for i in range(3):
                sb_id_codes.append(self._create_obs_sb(
                    self.ant_proxy, self.controlled, "VR.CM.AUTO.SC.50",
                    runtime=5))
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.hop("Assign newly created sbs {} to subarray {}".format(
                sb_id_codes, sub_nr))
            for sb_id in sb_id_codes:
                subarray.req.assign_schedule_block(sb_id)
            Aqf.hop("Scheduling sbs {}.".format(sb_id_codes))
            for sb_id in sb_id_codes:
                self.schedule_and_check_pass(sub_nr, sb_id)

            Aqf.step("4. Verify that the scheduling blocks do not start executing. ")
            Aqf.wait(2, "Give system chance to erroneously try to start execution")
            Aqf.hop('Check that schedule blocks are neither being executed nor completed')
            for sb in sb_id_codes:
                Aqf.is_false(sb in self.cam.exe.sensor.task_states.get_value(),
                             'Check that sb {} is not being executed'.format(sb))
                self.check_sb_state(sb, ScheduleBlockStates.SCHEDULED)
            Aqf.step("5. Manually execute a scheduling block and wait for completion. ")
            self.execute_and_check_pass(sub_nr, sb_id_codes[0])
            self.wait_sb_complete(sub_nr, sb_id_codes[0], timeout=300)
            self.check_sb_state(sb_id_codes[0], ScheduleBlockStates.COMPLETED)
            Aqf.step("6. Set the sub-array to Queue Scheduling Mode. ")
            Aqf.is_true(subarray.req.set_scheduler_mode("queue").succeeded,
                        'Request queue schedule mode via subarray_{} interface'
                    .format(sub_nr))
            Aqf.step("7. Verify that the ready scheduling blocks start execution through "
                     "queue scheduling and wait for completion. ")
            for sb_id in sb_id_codes[1:]:
                self.check_sb_running(sub_nr, sb_id)
                self.wait_sb_complete(sub_nr, sb_id, is_running_check=False, timeout=300)
                self.check_sb_state(sb_id, ScheduleBlockStates.COMPLETED)
            Aqf.step("8. Tear down the sub-array.")
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.SC.51')
    def test_cam_order_sbs(self):
        """
        Verify that the following can be done:
        1. Re-arrange/order the scheduling blocks in the sub-arrays observation
            schedule.
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        obs_sched_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))

        sb_id_codes = []
        try:
            subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
            Aqf.step("Create a sub-array. ")
            self.setup_default_array(sub_nr, activate=True)
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Setting scheduler to Manual Scheduling Mode")
            Aqf.hop('Activating subarray {}'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.step("Create and schedule three new schedule blocks")
            for i in range(3):
                sb_id_codes.append(self._create_obs_sb(
                    self.ant_proxy, self.controlled, "VR.CM.AUTO.SC.51",
                    runtime=5))
            Aqf.hop("Assign newly created sbs {} to subarray {}".format(
                sb_id_codes, sub_nr))
            for sb_id in sb_id_codes:
                subarray.req.assign_schedule_block(sb_id)
            Aqf.hop("Scheduling sbs {}.".format(sb_id_codes))
            for sb_id in sb_id_codes:
                self.schedule_and_check_pass(sub_nr, sb_id)

            last_sb = sb_id_codes[2]
            obs_sched = getattr(self.cam.sched.sensor, obs_sched_sensor_name).get_value()
            Aqf.step("Note the order of the SBs in the observation schedule are {}".format(obs_sched))
            Aqf.step("Setting priority of 3rd SB {} to HIGH".format(last_sb))
            reply = self.cam.sched.req.sb_set_priority(last_sb, 'HIGH')
            Aqf.wait(10, "Wait for update to be processed {}".format(reply))
            obs_sched = getattr(self.cam.sched.sensor, obs_sched_sensor_name).get_value()
            Aqf.step("Verify that high priority SB now at the top of the Observation Schedule {}".format(obs_sched))
            new_first_sb = obs_sched.split(",")[0]
            Aqf.equals(new_first_sb, last_sb, "Verify that high priority SB {} now at the top of the Observation Schedule {}".format(last_sb, obs_sched))

        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.SC.52')
    def test_cam_serial_sbs_in_queue_scheduling(self):
        """
        Verify that CAM serially executes schedule blocks in Queue Scheduling Mode:
        1. Set the sub-array to Manual Scheduling.
        2. Prepare the sub-array with some scheduling blocks in the Observation Schedule.
        3. Set the sub-array to Queue Scheduling.
        4. Verify that the ready scheduling blocks executes serially.
        5. Release the sub-array.
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        obs_sched_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))

        try:
            Aqf.step("Verify that CAM serially executes schedule blocks in Queue Scheduling Mode: ")
            Aqf.step("1. Set the sub-array to Manual Scheduling. ")
            Aqf.hop('Create subarray_{}'.format(sub_nr))
            self.setup_default_array(sub_nr, activate=True)
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        'Request manual schedule mode via subarray_{} interface'
                        .format(sub_nr))
            Aqf.hop('Clearing schedule for subarray_{}'.format(sub_nr))
            utils.clear_observation_schedule(self, sub_nr)
            Aqf.hop('Wait on sched.{} sensor to become '
                        'empty'.format(obs_sched_sensor_name))
            wait_sensor(self.cam.sched.sensor.observation_schedule_1, "", timeout=30)
            Aqf.step("2. Prepare the sub-array with some scheduling blocks in the "
                     "Observation Schedule.")
            Aqf.hop("Creating three new schedule blocks ")
            sb_id_codes = []
            for i in range(3):
                sb_id_codes.append(self._create_obs_sb(
                    self.ant_proxy, self.controlled, "VR.CM.AUTO.SC.50",
                    runtime=10))
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.hop("Scheduling sbs {}.".format(sb_id_codes))
            for sb_id in sb_id_codes:
                self.schedule_and_check_pass(sub_nr, sb_id)

            Aqf.step("3. Set the sub-array to Queue Scheduling. ")
            Aqf.is_true(subarray.req.set_scheduler_mode("queue").succeeded,
                        'Request queue schedule mode via subarray_{} interface'
                    .format(sub_nr))
            Aqf.step("4. Verify that the ready scheduling blocks executes serially. ")
            for sb_id in sb_id_codes:
                self.check_sb_running(sub_nr, sb_id)
                self.wait_sb_complete(sub_nr, sb_id, is_running_check=False, timeout=300)
                self.check_sb_state(sb_id, ScheduleBlockStates.COMPLETED)

            Aqf.step("5. Release the sub-array.")
            Aqf.is_true(subarray.req.free_subarray(timeout=30).succeeded,
                        'Request subarray_{} to free via {} interface'.format(
                            sub_nr, subarray.name))
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.SC.53')
    def test_cam_automatic_scheduling_modes(self):
        """
        Verify that CAM provides Automatic Scheduling Modes for each sub-array.
        (NOTE: Steps TBD as Automatic Scheduling Mode is only for Timescale C)
        """

        Aqf.step("Manual and queue scheduling is tested with VR.CM.AUTO.SC.50")
        Aqf.step("Verify that CAM provides Automatic Scheduling Mode for each sub-array. ")

        Aqf.waived("Automatic scheduling is only a Timescale C requirement")
        Aqf.end()


    @aqf_vr('VR.CM.AUTO.OBR.54')
    def test_cam_assign_sbs_to_subarray(self):
        """
        Test Schedule Block assignment to the sub-array as follows:
        1. Set up subarray
        2. Put sub-array into Manual Scheduling Mode.
        3. Using the Control Authority Interface on a sub-array, assign all SBs of a
           program block to the sub-array.
        4. Remove one SB from the sub-arrays observation schedule.
        5. Add that SB back to the sub-arrays observation schedule.
        6. Execute one of the SBs.
        7. Stop execution of the SB.
        8. Execute another one of the SBs from the sub-arrays observation schedule.
        9. Cancel execution of the SB.
        10. Release the sub-array.
        """

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        Aqf.step("Test Schedule Block assignment to the sub-array as follows: ")
        try:
            Aqf.step("1. Set up subarray.")
            Aqf.step("Using subarray=%s, receptor=%s, controlled=%s" % (
                sub_nr, self.ant_proxy, self.controlled))
            resource_str = ",".join([self.ant_proxy, self.controlled])
            subarr_spec = {"receptors": [self.ant_proxy], # list of receptor resource names
                "controlled": [self.controlled],
                "products": ["32K"], # CBF data product name (no array name)
                "band": "l"}
            self._create_subarray(sub_nr, subarr_spec)

            # Activating subarray
            Aqf.step("Activating subarray %s" % (sub_nr))
            subarray.req.activate_subarray(timeout=100)

            Aqf.step("2. Put sub-array into Manual Scheduling Mode. ")
            subarray.req.set_scheduler_mode("manual")

            Aqf.step("3. Using the Control Authority Interface on a sub-array, assign all "
                     "SBs of a program block to the sub-array. ")
            Aqf.step("4. Remove one SB from the sub-arrays observation schedule. ")
            Aqf.step("5. Add that SB back to the sub-arrays observation schedule. ")
            Aqf.step("6. Execute one of the SBs. ")
            Aqf.step("7. Stop execution of the SB. ")
            Aqf.step("8. Execute another one of the SBs from the sub-arrays observation "
                     "schedule.")
            Aqf.step("9. Cancel execution of the SB.")
            Aqf.step("10. Release the sub-array.")
            Aqf.step("TBD - program block support to be added later (TS C&D with observation preparation and proposal management)")
            # Aqf.failed("TBD")
        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('CAM_basic_script')
    def test_cam_basic_script(self):

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        obs_sched_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        try:
            Aqf.step("1. Create a sub-array. ")
            Aqf.hop('Create subarray_{}'.format(sub_nr))
            self.setup_default_array(sub_nr, activate=False)
            Aqf.step("2. Activate the sub-array. ")
            Aqf.is_true(subarray.req.activate_subarray(timeout=100).succeeded,
                        "Activation request for subarray {} successful".format(sub_nr))
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.step("3. Create a basic-script")
            sb_id = self._create_obs_sb(self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10)
            Aqf.step("4. Set scheduler to manual")
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")
            Aqf.hop("5. Assign newly created sb %s to subarray %s" % (sb_id, sub_nr))
            subarray.req.assign_schedule_block(sb_id)
            Aqf.hop('Schedule sb {}'.format(sb_id))
            self.schedule_and_check_pass(sub_nr, sb_id)
            Aqf.hop('Execute sb {}'.format(sb_id))
            self.execute_and_check_pass(sub_nr, sb_id)
            self.wait_sb_complete(sub_nr, sb_id, timeout=300)
            now = datetime.utcnow()
	    controlled_line = self.check_progress_log_for_string(sb_id, 'Controlled objects: ')
            # progress_filename = "/var/kat/tasklog/%02d/%02d/%02d/%s/progress.out" % (
            #     now.year, now.month, now.day, sb_id)
            # with open(progress_filename) as f:
            #     controlled_line = [l.strip() for l in f.readlines()
            #                        if l.startswith('Controlled objects: [')]
            # Will be done below in the finally statement
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    def get_component_from_cam(self, comp_name):
        counter = 20
        service = None
        while counter > 0:
            counter -= 1
            service = getattr(self.cam, comp_name, None)
            if service:
                counter = -1
            else:
                time.sleep(1)
        return service

    @aqf_vr('CAM_basic_script_with_set_band')
    def test_cam_basic_script_with_set_band(self):

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        obs_sched_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        # subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        subarray = self.get_component_from_cam('subarray_{}'.format(sub_nr))
        try:
            if "kat7" not in self.cam.katconfig.site:
                mcp = self.get_component_from_cam("mcp")
                Aqf.hop("cmc_array_list() {}".format(mcp.req.cmc_array_list()))
                Aqf.sensor(mcp.sensor.cmc_state).wait_until('synced', sleep=1, counter=1)
            Aqf.step("Free subarray")
            subarray.req.free_subarray(timeout=30)
            Aqf.step("1. Create a sub-array. ")
            Aqf.hop('Create subarray_{}'.format(sub_nr))
            self.setup_default_array(sub_nr, activate=False)
            Aqf.step("2. Set band, then activate the sub-array. ")
            result = subarray.req.set_band("l")
            Aqf.is_true(result.succeeded,
                        "Verify set_band for l-band on subarray {} succeeded".format(sub_nr))
            Aqf.step("Verify all resources are synced...")
            verify_sync_results = [
                r.state == 'synced'
                    for r in subarray.children.values()]

            if not all(verify_sync_results):
                Aqf.wait(10, "Subarray {} resources not synced, waiting 10 seconds".format(sub_nr))
                verify_sync_results = [
                    r.state == 'synced'
                        for r in subarray.children.values()]

            if not all(verify_sync_results):
                not_synced_resources = [
                    r.state != 'synced'
                        for r in subarray.children.values()]

                msg = "Subarray {} resources {} is not synced, aborting test".format(sub_nr, not_synced_resources)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            result = subarray.req.activate_subarray(timeout=180)
            Aqf.is_true(result.succeeded,
                        "Verify activate_subarray request for subarray {} succeeded".
                        format(sub_nr))
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until(
                "active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.step("3. Create a basic-script")
            sb_id = self._create_obs_sb(self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10)
            Aqf.step("4. Set scheduler to manual")
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")
            Aqf.hop("5. Assign newly created sb %s to subarray %s" % (sb_id, sub_nr))
            subarray.req.assign_schedule_block(sb_id)
            Aqf.hop('Schedule sb {}'.format(sb_id))
            self.schedule_and_check_pass(sub_nr, sb_id)
            Aqf.hop('Execute sb {}'.format(sb_id))
            self.execute_and_check_pass(sub_nr, sb_id)
            self.wait_sb_complete(sub_nr, sb_id, timeout=300)
            # now = datetime.utcnow()
            # progress_filename = ("/var/kat/tasklog/%02d/%02d/%02d/%s/"
            #                      "progress.out" %
            #                      (now.year, now.month, now.day, sb_id))
            # with open(progress_filename) as f:
            #     controlled_line = [l.strip() for l in f.readlines()
            #                        if l.startswith('Controlled objects: [')]
            # Will be done below in the finally statement
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

        Aqf.end()

    @aqf_vr('CAM_basic_script_with_set_band_and_product')
    def _test_cam_basic_script_with_set_band_and_product(self):

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        obs_sched_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))

        try:
            Aqf.step("1. Create a sub-array. ")
            Aqf.hop('Create subarray_{}'.format(sub_nr))
            self.setup_default_array(sub_nr, activate=False)
            Aqf.step("2. Select band and product, then activate the sub-array.")
            Aqf.is_true(subarray.req.set_band("l").succeeded,
                        "Set l band for subarray {} successful".format(sub_nr))
            Aqf.is_true(subarray.req.set_product("c856M4k", 1.0).succeeded,
                        "Set product for subarray {} successful".format(sub_nr))
            Aqf.is_true(subarray.req.activate_subarray(timeout=100).succeeded,
                        "Verify activate_subarray successful for subarray {}".
                        format(sub_nr))
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until(
                "active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.step("3. Create a basic-script")
            sb_id = self._create_obs_sb(self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10)
            Aqf.step("4. Set scheduler to manual")
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")
            Aqf.hop("5. Assign newly created sb %s to subarray %s" %
                    (sb_id, sub_nr))
            subarray.req.assign_schedule_block(sb_id)
            Aqf.hop('Schedule sb {}'.format(sb_id))
            self.schedule_and_check_pass(sub_nr, sb_id)
            Aqf.hop('Execute sb {}'.format(sb_id))
            self.execute_and_check_pass(sub_nr, sb_id)
            self.wait_sb_complete(sub_nr, sb_id, timeout=300)
	    controlled_line = self.check_progress_log_for_string(sb_id, 'Controlled objects: ')
            # now = datetime.utcnow()
            # progress_filename = "/var/kat/tasklog/%02d/%02d/%02d/%s/progress.out" % (
            #     now.year, now.month, now.day, sb_id)
            # with open(progress_filename) as f:
            #     controlled_line = [l.strip() for l in f.readlines()
            #                        if l.startswith('Controlled objects: [')]
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()

    @aqf_vr('CAM_basic_session_track')
    def test_cam_basic_session_track(self):

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")

        sub_nr = random.randint(1, 4)
        obs_sched_sensor_name = 'observation_schedule_{}'.format(sub_nr)
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))

        try:
            Aqf.is_true(subarray.req.free_subarray(timeout=30).succeeded,
                'Request subarray_{} to free via {} interface'.format(
                sub_nr, subarray.name))
            Aqf.step("1. Create a sub-array. ")
            Aqf.hop('Create subarray_{}'.format(sub_nr))
            self.setup_default_array(sub_nr, activate=False)
            Aqf.step("2. Select band and product, then activate the sub-array.")
            Aqf.is_true(subarray.req.set_band("l").succeeded,
                        "Set l band for subarray {} successful".format(sub_nr))
            Aqf.is_true(subarray.req.set_product("c856M4k", 1.0).succeeded,
                        "Set product for subarray {} successful".format(sub_nr))
            Aqf.is_true(subarray.req.activate_subarray(timeout=100).succeeded,
                        "Verify activate_subarray successful for subarray {}".
                        format(sub_nr))
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until(
                "active", sleep=1, counter=5)
            if not ok:
                msg = "Subarray {} is not active, aborting test".format(sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            # Create basic-session-track SB
            Aqf.step("3. Create a basic-session-track")
            description = "Basic Session Track for %s" % self.ant_proxy
            instruction_set = (
                    "run-obs-script ~/svn/katscripts/cam/basic-session-track.py azel,20,30 -t 10 -n off ")
            sb_id = self._create_obs_sb(self.ant_proxy, self.controlled,
                                        "CAM_basic_session_track", runtime=10,
                                        description=description,
                                        instruction_set=instruction_set)

            Aqf.step("4. Set scheduler to manual")
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")
            Aqf.hop("5. Assign newly created sb %s to subarray %s" %
                    (sb_id, sub_nr))
            subarray.req.assign_schedule_block(sb_id)

            Aqf.hop('Schedule sb {}'.format(sb_id))
            ok = self.schedule_and_check_pass(sub_nr, sb_id)
            if not ok:
                msg = "SB {} on subarray {} was not scheduled, aborting test".format(sb_id, sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            Aqf.hop('Execute sb {}'.format(sb_id))
            ok = self.execute_and_check_pass(sub_nr, sb_id)
            if not ok:
                msg = "SB {} on subarray {} execute failed, aborting test".format(sb_id, sub_nr)
                Aqf.failed(msg)
                Aqf.end()
                raise RuntimeError(msg)

            ok = self.wait_sb_complete(sub_nr, sb_id, timeout=300)
	    controlled_line = self.check_progress_log_for_string(sb_id, 'Controlled objects: ')
            # now = datetime.utcnow()
            # progress_filename = "/var/kat/tasklog/%02d/%02d/%02d/%s/progress.out" % (
            #     now.year, now.month, now.day, sb_id)
            # with open(progress_filename) as f:
            #     controlled_line = [l.strip() for l in f.readlines()
            #                        if l.startswith('Controlled objects: [')]
        finally:
            try:
                utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
            finally:
                Aqf.end()
