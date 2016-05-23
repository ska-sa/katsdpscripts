###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
from tests import settings, specifics, wait_sensor_includes
from tests import wait_sensor_includes, wait_sensor_excludes, Aqf, AqfTestCase

from nosekatreport import (aqf_vr, system, slow, site_acceptance)

from datetime import datetime
import json
import os
import time
import traceback

import katcorelib
from katuilib import ScheduleBlockStates, ScheduleBlockTypes, ScheduleBlockPriorities
import katconf
from katconf.sysconf import KatObsConfig
from katmisc.utils import csv_sets
from katmisc.utils.sensor_parsing import (
    resources_from_allocations)
from katobs.katexecutor import katexecutor
from katcorelib.katobslib.test import test_manager
from katobs.katexecutor import katexecutor
from tests import utils


##############################################################################
### TBD - this file should be reworked for Aqf and for all systems
### as it tests core design concepts of the scheduling and resource allocation
##############################################################################


@system('all')
class TestObsControl(AqfTestCase):

    """Tests for mkat_rts observation control."""

    def set_up_db(self, params_list):

        for params in params_list:
            resource_spec = self.db_manager.create_resource_spec(
                    params.get('ants', 'available'),
                    params.get('controlled',self.controlled))
            schedule_block = self.db_manager.create_schedule_block(
                    resource_spec=resource_spec.id,
                    owner=params.get('owner', None),
                    sb_type=params.get('type', ScheduleBlockTypes.OBSERVATION),
                    priority=params.get('priority', ScheduleBlockPriorities.LOW),
                    desired_start_time=params.get('desired_start_time', None),
                    expected_duration_seconds=params.get('expected_duration_seconds', None),
                    instruction_set=params.get('instruction_set', None))
            self.sb_id_codes.append(schedule_block.id_code)

    def setUp(self):
        self.sb_id_codes = []
        config = KatObsConfig(self.cam.system)
        # Create a katobs database manager for this system configuration.
        self.db_manager = test_manager.KatObsDbTestManager(config.db_uri)
        utils.clear_all_subarrays_and_schedules(self)
        # Set scheduler to manual mode.
        self.cam.sched.req.mode(1,'manual')
        self.controlled = specifics.get_controlled_data_proxy(self)

    def tearDown(self):
        # Remove what tests added to the database.
        pass

    def check_verification_state(self, id_code, verification_state):
        """Test verification result of schedule block."""
        # Get schedule block from db
        self.db_manager.expire()
        schedule_block = self.db_manager.get_schedule_block(id_code)
        # Check not verified
        # self.assertEqual(schedule_block.verification_state, verification_state)
        Aqf.equals(schedule_block.verification_state, verification_state,
                    'Check verification state')
        Aqf.passed('Schedule block %s verification state is %s.' %
                   (id_code, verification_state.key))

    def check_task_log(self, id_code, is_dry_run):
        """Check that outfile for task has been created.

        Parameters
        ----------
        id_code : str
            Schedule block ID code.
        is_dry_run : bool
            True if for dry-run, False if for normal task execution."""

        now = datetime.utcnow()

        outfile = "/var/kat/tasklog/%02d/%02d/%02d/%s/%s" % \
                (now.year, now.month, now.day, id_code,
                        "dryrun.out" if is_dry_run else "progress.out")
        # self.assertEqual(os.path.isfile(outfile), True)
        Aqf.equals(os.path.isfile(outfile), True, 'Check the progress')
        Aqf.passed('Schedule block %s has dry-run outfile %s.' % \
                (id_code, outfile))

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
        reply = self.cam.sched.req.sb_verify(1,id_code)
        # self.assertEqual(reply.succeeded, success)
        Aqf.equals(reply.succeeded, success, 'Check the request verification')
        Aqf.passed('Schedule block %s verification request succeeded.' % id_code)
        if success:
            #Wait for verification to completed
            ok = wait_sensor_includes(self.cam, self.cam.exe.sensor.dryrun_task_states, "%s:%s" % (id_code, task_state), 180)
            # self.assertTrue(ok, 'cam.exe dryrun_task_states does not contain %s:%s' % (id_code, task_state))
            Aqf.equals(ok, True, 'Verify cam.exe dryrun_task_states includes %s:%s' % (id_code, task_state))
            Aqf.passed('Schedule block %s verification %s.' % (id_code, task_state))
        time.sleep(5)
        self.check_verification_state(id_code, verification_state)

    def check_verification_declined(self, id_code):
        """Test verification result of schedule block."""
        # Request verification
        reply = self.cam.sched.req.sb_verify(1,id_code)
        # Check verified: ok
        self.assertEqual(reply.succeeded, False)
        Aqf.passed('Schedule block %s verification declined.' % id_code)

    def schedule_and_check_pass(self, sb_id_code):
        """Test for success of scheduling a schedule block."""
        all_ok = True
        reply = self.cam.sched.req.sb_schedule(1, sb_id_code)
        all_ok = all_ok and Aqf.is_true(reply.succeeded, 'Verify that schedule request succeeded for schedule block %s.' % sb_id_code)

        self.db_manager.expire()
        schedule_block = self.db_manager.get_schedule_block(sb_id_code)
        if schedule_block.type == ScheduleBlockTypes.OBSERVATION:
            #Wait for verify to complete
            ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.observation_schedule_1, sb_id_code, 180)
            all_ok = all_ok and Aqf.is_true(ok, 'Verify cam.sched observation_schedule_1 includes %s' % (sb_id_code))
            self.db_manager.expire()
            schedule_block = self.db_manager.get_schedule_block(sb_id_code)
            all_ok = all_ok and Aqf.equals(schedule_block.verification_state, ScheduleBlockVerificationStates.VERIFIED,
                       'Verify schedule block %s is VERIFIED.' % sb_id_code)

        #Wait for SB to be scheduled
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.observation_schedule_1, sb_id_code, 180)
        all_ok = all_ok and Aqf.is_true(ok, 'Verify cam.sched observation_schedule_1 includes %s' % (sb_id_code))
        self.db_manager.expire()
        schedule_block = self.db_manager.get_schedule_block(sb_id_code)
        all_ok = all_ok and Aqf.equals(schedule_block.state, ScheduleBlockStates.SCHEDULED,
                   'Verify schedule block %s is SCHEDULED.' % sb_id_code)
        return all_ok

    def execute_and_check_pass(self, sb_id_code, type):
        """Test for success of executing a schedule block."""
        all_ok = True
        all_ok = all_ok and Aqf.passed('Verify schedule block %s can be executed' %
                   sb_id_code)
        time.sleep(3.0)
        result = self.cam.sched.req.sb_execute(1,sb_id_code)
        if result.succeeded:
            Aqf.passed('Execute request successful schedule block %s' %
                       sb_id_code)
        else:
            all_ok = False
            Aqf.failed('Shedule block %s could not be executed' %
                       sb_id_code)

        #Check to see that SB state is ACTIVE
        Aqf.step('Verify that %s is ACTIVE' % (sb_id_code))
        ok = wait_sensor_includes(self.cam, self.cam.sched.sensor.active_schedule_1,
                                    sb_id_code, 240)
        all_ok = all_ok and Aqf.is_true(ok, 'Verify cam.sched.sensor.active_schedule_1 includes %s' %
                    (sb_id_code))

        allocations = []
        allocations_str = self.cam.katpool.sensor.allocations.get_value()
        if not allocations_str == '':
            allocations = [allocation[0] for allocation in
                           json.loads(allocations_str)
                           if allocation[1] == sb_id_code]
        all_ok = all_ok and Aqf.is_true(len(allocations) > 0,
                    'Check that allocation list is more than 0')
        Aqf.passed('Schedule block %s has resources allocated to it' %
                   sb_id_code)

        if type == ScheduleBlockTypes.OBSERVATION:
            task_states_string = self.cam.exe.sensor.task_states.get_value()
            task_states = dict([item.split(":") for item in
                                csv_sets.set_from_csv(task_states_string)])
            self.assertEqual(task_states[sb_id_code],
                             katexecutor.TaskStates.RUNNING.key)
            self.check_task_log(sb_id_code, False)
            Aqf.passed('Schedule block %s is being executed by katexecutor.' %
                       sb_id_code)

        return all_ok

    def execute_and_check_fail(self, sb_id_code):
        """Test for failure of executing a schedule block."""
        Aqf.step('Verify that schedule block %s cannot be executed' % sb_id_code)
        result = self.cam.sched.req.sb_execute(1,sb_id_code)
        if not result.succeeded:
            Aqf.passed('Not executing schedule block %s' % sb_id_code)
        else:
            Aqf.failed('Schedule block %s should not have been executed' % sb_id_code)

    def _setup_subarray_and_verify(self, sub_nr):

        controlled = specifics.get_controlled_data_proxy(self, sub_nr=sub_nr)
        selected_ants_set = utils.setup_subarray(self, sub_nr, controlled)

        if not selected_ants_set:
            Aqf.failed("Resource assignment to subarray_{} failed. Aborting test".format(sub_nr))
            return False

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
            return False

        if self.cam.sched.sensor.mode_1.get_value() == 'locked':
            Aqf.failed("Scheduler mode is 'locked'. Aborting test")
            return False

        return selected_ant

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
        sb_id_code = self.obs.sb.new(owner=user, antenna_spec='m000')
        return sb_id_code

    @aqf_vr('VR.CM.AUTO.OBR.8')
    def test_resources_assigned_to_single_subarray(self):
        """Test CAM allows a resource to be assigned to a single subarray"""

        sub_nr = 1
        sub_nr_2 = 2
        subarray_1 = "subarray_%s" % sub_nr
        subarray_2 = "subarray_%s" % sub_nr_2
        ant_1 = self.cam.ants[0].name
        ant_2 = self.cam.ants[1].name
        Aqf.step("Using subarrays=%s,%s, receptors=%s,%s" % (
            sub_nr, sub_nr_2, ant_1, ant_2))
        sub1 = getattr(self.cam, subarray_1)
        sub2 = getattr(self.cam, subarray_2)

        try: # Try the test

            reply = sub1.req.assign_resources(ant_1)
            Aqf.equals(reply.succeeded, True, "Verify that assigning {} to {} succeeded".format(ant_1,subarray_1))
            reply = sub2.req.assign_resources(ant_1)
            Aqf.equals(reply.succeeded, False, "Verify that assigning {} to another {} fails - as expected"
                      .format(ant_1,subarray_2))
            Aqf.is_true(ant_1 in self.cam.katpool.sensor.pool_resources_1.get_value(),
                      "Verify that {} is in {} resource pool".format(ant_1,subarray_1))
            Aqf.is_true(ant_1 not in self.cam.katpool.sensor.pool_resources_2.get_value(),
                      "Verify that {} is NOT in {} resource pool - as expected".format(ant_1,subarray_2))

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
            raise
        finally:
            try:
                Aqf.step('Cleanup - Free subarray_1')
                sub1.req.free_subarray()
            finally:
                Aqf.end()

    @aqf_vr('VR.CM.AUTO.OBR.8')
    def test_exclusive_resource_allocation_to_SB(self):
        """Tests for single allocation of a resource assigned to a subarray - using manual SBs."""

        ant_first = self.cam.ants[0].name
        ant_second = self.cam.ants[1].name
        
        Aqf.step("Assign selected resources to subarray_1 - {},{},{}."
                 .format(self.controlled, ant_first, ant_second))
        res_csv = ",".join([ant_first, ant_second, self.controlled])
        self.cam.subarray_1.req.assign_resources(res_csv)

        SB_PARAMS_LIST = [
                {
                    'ants': ant_first,
                    'controlled': self.controlled,
                    'owner': 'aqf-test',
                    'type': ScheduleBlockTypes.MANUAL,
                    },
                {
                    'ants': '%s,%s' % (ant_first, ant_second) ,
                    'controlled': '',
                    'owner': 'aqf-test',
                    'type': ScheduleBlockTypes.MANUAL,
                    },
                {
                    'ants': ant_second,
                    'controlled': self.controlled,
                    'owner': 'aqf-test',
                    'type': ScheduleBlockTypes.MANUAL,
                    },
                ]

        try:
            Aqf.step("Add schedule blocks, resource specs for this test to katobs db - "
                     "using ants {},{} and {}".format(ant_first, ant_second, self.controlled))
            self.set_up_db(SB_PARAMS_LIST)

            Aqf.step("Set up manual schedule blocks that compete for resources - using {},{} and {}."
                 .format(ant_first, ant_second, self.controlled))
            Aqf.step("SB {} requires {} and {}"
                 .format(self.sb_id_codes[0], ant_first, self.controlled))
            Aqf.step("SB {} requires {},{}"
                 .format(self.sb_id_codes[1], ant_first, ant_second))
            Aqf.step("SB {} requires {} and {}"
                 .format(self.sb_id_codes[2], ant_second, self.controlled))

            # Scheduling all schedule blocks should pass
            for id_code in self.sb_id_codes:
                self.schedule_and_check_pass(id_code)
            # Executing the first manual schedule block - it should pass
            Aqf.step('Verify that first manual schedule block {} starts executing with {}'
                     .format(self.sb_id_codes[0], ant_first))
            self.execute_and_check_pass(self.sb_id_codes[0], ScheduleBlockTypes.MANUAL)

            # Executing the 2nd and 3rd (competing) schedule blocks should fail
            for id_code in self.sb_id_codes[1:]:
                Aqf.step('Verify that schedule block {} competing for {},{},{} cannot be executed'
                     .format(id_code, self.controlled, ant_first, ant_second))
                self.execute_and_check_fail(id_code)

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))

        finally:
            try:
                #Try and cleanup
                Aqf.step('Cleanup - Complete the executing SB, and move others to DRAFT')
                self.cam.sched.req.sb_complete(1,self.sb_id_codes[0])
                self.cam.sched.req.sb_to_draft(1,self.sb_id_codes[1])
                self.cam.sched.req.sb_to_draft(1,self.sb_id_codes[2])
                Aqf.wait(5, "Cleanup - Waiting for SB to be completed")
                Aqf.step('Cleanup - Free subarray_1')
                self.cam.subarray_1.req.free_subarray()
            finally:
                Aqf.end()

