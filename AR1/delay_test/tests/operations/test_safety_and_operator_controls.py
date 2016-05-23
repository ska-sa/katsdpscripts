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

import katuilib

from datetime import datetime

from tests import utils, specifics, settings, wait_sensor, Aqf, AqfTestCase
from nosekatreport import (aqf_vr, system, slow, site_acceptance)

from katconf.sysconf import KatObsConfig
from katmisc.utils import csv_sets
from katmisc.utils.sensor_parsing import resources_from_allocations
from katobs.katexecutor import katexecutor
from katcorelib.katobslib.test import test_manager
from katuilib import (ScheduleBlockStates, ScheduleBlockTypes,
                      ScheduleBlockPriorities)

##############################################################################
### TBD - this file should be reworked for Aqf and for all systems
### as it tests core design concepts of the scheduling and resource allocation
##############################################################################

@system('all')
class TestObsControl(AqfTestCase):

    """Tests for mkat_rts observation control."""


    def setUp(self):
        self.schedule_block_id_codes = []
        config = KatObsConfig(self.cam.system)
        # Create a katobs database manager for this system configuration.
        self.db_manager = test_manager.KatObsDbTestManager(config.db_uri)
        # Set scheduler 1 to manual mode.
        self.cam.sched.req.mode(1,'manual')

    def tearDown(self):
        # Remove what tests added to the database.
        pass

    def _point_antennas(self, az, el):
        """Point antennas to given az and el."""
        Aqf.step("Point the antennas to make them move")
        az = float(az)
        el = float(el)
        Aqf.step("Verify that antennas are pointing")
        for ant in self.cam.ants:
            Aqf.sensor(ant.sensor.mode).wait_until('POINT', sleep=1)
            Aqf.equals('POINT', ant.sensor.mode.get_value(),
                       'Check that antenna %s is pointing' % ant.name)

    def _check_observations_stopped(self, sub_nr):
        sens_obj = getattr(self.cam.sched.sensor, "mode_%s" % sub_nr)
        sched_mode = sens_obj.get_value()
        Aqf.equals('locked', sched_mode, 'Scheduler %s is locked' % sub_nr)

    def _check_no_running_observations(self, sub_nr):
        count = 0
        ok = False
        sens_obj = getattr(self.cam.sched.sensor, "active_schedule_%s" % sub_nr)
        while count < 5 and not ok:
            Aqf.wait(2, "Wait for observations to complete on %s" % sub_nr)
            active_schedule = sens_obj.get_value()
            ok = active_schedule == ""
            count += 1
        Aqf.is_true(ok, "Check that no observations are running on %s" % sub_nr)

    def _check_antennas_stowed(self):
        for ant in self.cam.ants:
            mode = ant.sensor.mode.get_value()
            Aqf.equals(mode, "STOW", 'Check %s is in STOW' % ant.name)

    def _check_antennas_stopped(self):
        for ant in self.cam.ants:
            mode = ant.sensor.mode.get_value()
            Aqf.equals(mode, "STOP", 'Check %s is in STOP' % ant.name)

    def _check_antennas_inhibited(self):
        for ant in self.cam.ants:
            inhibited = ant.sensor.inhibited.get_value()
            Aqf.equals(inhibited, True, 'Check %s is inhibited' % ant.name)

    def _check_antennas_ready(self):
        for ant in self.cam.ants:
            mode = ant.sensor.mode.get_value()
            inhibited = ant.sensor.inhibited.get_value()
            Aqf.equals("STOP", mode, 'Verify %s is in STOP (not stowed)' % ant.name)
            Aqf.equals(inhibited, False, 'Verify %s is not inhibited' % ant.name)

    def _create_obs_sb(self, selected_ant, program_block):
        controlled = "data"
        sb_id_code = self.obs.sb.new(owner="aqf-test", antenna_spec=selected_ant, controlled_resources=controlled)
        self.obs.sb.description = "Track for %s" % selected_ant
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        self.obs.sb.instruction_set = "run-obs-script ~/scripts/observation/track.py -t 60.0 -m 300 --no-delays --repeat --proposal-id=CAM_AQF --program-block-id=%s azel,45,50" % program_block
        Aqf.wait(1)
        return sb_id_code

    def _setup_subarray_and_verify(self, sub_nr):
        controlled = "data"
        specific_controlled = specifics.get_specific_controlled(self, sub_nr, controlled)
        selected_ants_set = utils.setup_subarray(self, sub_nr, controlled)

        if not selected_ants_set:
            Aqf.failed("Resource assignment to subarray_{} failed. Aborting test".format(sub_nr))
            return False

        selected_ant = list(selected_ants_set)[0] # Choose one of the ants
        Aqf.step("Select resources for the test - {} and {}"
                 .format(selected_ant, specific_controlled))
        assignments = self.cam.katpool.sensor.pool_resources_1.get_value()
        if selected_ant in assignments and specific_controlled in assignments:
            Aqf.step("Selected resources are assigned - {} and {}"
                     .format(selected_ant, specific_controlled))
        else:
            Aqf.failed("Selected resources NOT assigned - {} and {} - assignments {}. Aborting test"
                       .format(selected_ants_set, specific_controlled, assignments))
            return False

        if self.cam.sched.sensor.mode_1.get_value() == 'locked':
            Aqf.failed("Scheduler mode is 'locked'. Aborting test")
            return False

        Aqf.step("Activate subarray_1")
        self.cam.subarray_1.req.activate_subarray()

        return selected_ant

    @aqf_vr("VR.CM.AUTO.I.2")
    def test_safety_control_operator_stow_all_ants(self):
        """Test actions when operator request to stow all antennas."""

        def teardown_subarray():
            utils.teardown_subarray(self, 1)

        sub_nr = 1
        selected_ant = self._setup_subarray_and_verify(sub_nr)
        if not selected_ant:
            teardown_subarray()
            Aqf.end()
            return

        try: #Try the test
            # Create an OBSERVATION schedule block
            Aqf.step("Create an OBSERVATION schedule block")
            sb_id_code = self._create_obs_sb(selected_ant, "VR.CM.AUTO.I.2-stow-all")

            Aqf.step("Assign and schedule OBSERVATION SB %s" % sb_id_code)
            self.cam.subarray_1.req.assign_schedule_block(sb_id_code)
            self.cam.subarray_1.req.sb_schedule(sb_id_code)
            count = 30
            scheduled = False
            while (count > 0 and not scheduled):
                count -= 1
                observation_schedule = self.cam.sched.sensor.observation_schedule_1.get_value()
                scheduled = sb_id_code in observation_schedule
                Aqf.wait(2, "Wating for SB {} to verify and land in observation schedule '{}'".format(sb_id_code, observation_schedule))
            if not scheduled:
                Aqf.failed("OBSERVATION SB %s was not scheduled. Exiting test." % sb_id_code)
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SB %s was scheduled" % sb_id_code)

            Aqf.step("Activate OBSERVATION SB %s" % sb_id_code)
            self.cam.subarray_1.req.sb_execute(sb_id_code, timeout=10)
            count = 30
            active = False
            while (count > 0 and not active):
                count -= 1
                active_schedule = self.cam.sched.sensor.active_schedule_1.get_value()
                active = sb_id_code in active_schedule
                Aqf.wait(2, "Waiting for SB {} to verify and execute and land in active_schedule '{}'".format(sb_id_code, active_schedule))
            if not active:
                Aqf.failed("OBSERVATION SB %s was not activated. Exiting "
                           "test." % sb_id_code)
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SB %s was activated" % sb_id_code)

            # Allowing observation to run for few seconds before we stow ants
            Aqf.step("Allow OBSERVATION SB %s to run for a few seconds before "
                     "operator intervenes" % sb_id_code)
            Aqf.wait(15)

            Aqf.step("Inject operator request to stow all antennas")
            self.cam.sys.req.operator_stow_antennas()
            Aqf.wait(3, "Wating for operator input to be processed")

            Aqf.step("Verifying that the necessary actions were successful")
            """
            Verification Requirements:
            --------------------------
            Test that the CAM implements the following
            upon receiving an intervention to stow:

            1. (REMOVED)Interrupt executing SBs
            2. (REMOVED)Stop schedulers
            3. Stow (and inhibit) antennas
            4. Test that an attempt to point an antenna fails.
            """

            Aqf.step("Verify that antennas are inhibited")
            self._check_antennas_inhibited()
            Aqf.step("Verify that antennas are STOWED")
            self._check_antennas_stowed()
            Aqf.step("Verifying that antennas cannot be pointed")
            Aqf.step("Set antenna targets")
            self.cam.ants.req.target_azel(77.7, 66.6)
            Aqf.step("Set antennas to POINT")
            self.cam.ants.req.mode("POINT")
            Aqf.hop("Verify that antennas are not pointing but still in STOWED")
            self._check_antennas_stowed()
            Aqf.step("Verify that Scheduler is ready (not locked)")
            ready = utils.check_scheduler_ready(self, sub_nr)
            if ready:
                Aqf.passed("Scheduler %s is ready - not mode 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % sub_nr)

            #Test cleanup
            Aqf.step("---Test cleanup---")
            #Resume operations
            Aqf.step("Resume operations")
            self.cam.sys.req.operator_resume_operations()
            Aqf.wait(3, "Wait for operations to resume")
            Aqf.step("Verify that Scheduler is ready (not locked)")
            ready = utils.check_scheduler_ready(self, sub_nr)
            if ready:
                Aqf.passed("Scheduler %s is ready - not mode 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % sub_nr)
            Aqf.step("Verify that antennas are ready to continue ")
            self._check_antennas_ready()

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
            try:
                #Try to cleanup - Resume operations
                Aqf.step("Resume operations")
                self.cam.sys.req.operator_resume_operations()
            except:
                pass
            Aqf.hop('Cleanup - Mark the SB as complete. Only really needed if '
                    'the test failed.')
            self.cam.subarray_1.req.sb_complete(sb_id_code, True)

        finally:
            teardown_subarray()

        Aqf.end()


    @site_acceptance
    @aqf_vr("VR.CM.AUTO.I.2")
    def test_safety_control_operator_inhibit_all_ants(self):
        """Test actions when operator request to inhibit all antennas."""

        def teardown_subarray():
            utils.teardown_subarray(self, 1)

        sub_nr = 1
        selected_ant = self._setup_subarray_and_verify(sub_nr)
        if not selected_ant:
            teardown_subarray()
            Aqf.end()
            return

        try: #Try the test
            # Create an OBSERVATION schedule block
            Aqf.step("Create an OBSERVATION schedule block")
            sb_id_code = self._create_obs_sb(selected_ant, "VR.CM.AUTO.I.2-inhibit-all")
            
            Aqf.step("Assign and schedule OBSERVATION SB %s" % sb_id_code)
            self.cam.subarray_1.req.assign_schedule_block(sb_id_code)
            self.cam.subarray_1.req.sb_schedule(sb_id_code)
            count = 30
            scheduled = False
            while (count > 0 and not scheduled):
                count -= 1
                observation_schedule = self.cam.sched.sensor.observation_schedule_1.get_value()
                scheduled = sb_id_code in observation_schedule
                Aqf.wait(2, "Waiting for SB {} to verify and land in observation schedule '{}'".format(sb_id_code, observation_schedule))
            if not scheduled:
                Aqf.failed("OBSERVATION SB %s was not scheduled. Exiting test." % sb_id_code)
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SB %s was scheduled." % sb_id_code)

            Aqf.step("Activate OBSERVATION SB %s" % sb_id_code)
            self.cam.subarray_1.req.sb_execute(sb_id_code, timeout=10)
            count = 30
            active = False
            while (count > 0 and not active):
                count -= 1
                active_schedule = self.cam.sched.sensor.active_schedule_1.get_value()
                active = sb_id_code in active_schedule
                Aqf.wait(2, "Waiting for SB {} to verify and land in active schedule '{}'".format(sb_id_code, active_schedule))
            if not active:
                Aqf.failed("OBSERVATION SB %s was not activated. Exiting "
                           "test." % sb_id_code)
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SB %s was activated." % sb_id_code)

            # Allowing observation to run for few seconds before we stow ants
            Aqf.step("Allow OBSERVATION SB %s to run for a while "
                     "before operator intervenes" % sb_id_code)
            Aqf.wait(15)

            Aqf.step("Inject operator request to inhibit all antennas")
            self.cam.sys.req.operator_inhibit_antennas()
            Aqf.wait(3, "Wait for operator input to be processed")
            Aqf.step("Verifying that the necessary actions were successful")
            """
            Verification Requirements:
            --------------------------
            Test that the CAM implements the following
            upon receiving an intervention to stow:

            1. Interrupt executing SBs
            2. Stop schedulers
            3. Stop antenna movement immediately (don't first stow)
            4. Test that an attempt to point an antenna fails.
            """
            Aqf.step("Verify that observations are stopped (Scheduler 1 mode is 'locked')")
            self._check_observations_stopped(1)
            Aqf.step("Verify that no observations are running on subarray 1")
            self._check_no_running_observations(1)
            Aqf.step("Verify that Scheduler is LOCKED")
            locked = utils.check_scheduler_locked(self, sub_nr)
            if locked:
                Aqf.passed("Scheduler %s mode is 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s mode not 'locked'" % sub_nr)
            
            Aqf.step("Verify that antennas are inhibited")
            self._check_antennas_inhibited()

            Aqf.step("Verify that antennas are STOPPED")
            self._check_antennas_stopped()

            Aqf.step("Verifying that antennas cannot be pointed")
            Aqf.step("Set antenna targets")
            self.cam.ants.req.target_azel(77.7, 66.6)
            Aqf.step("Set antennas to POINT")
            self.cam.ants.req.mode("POINT")

            Aqf.step("Verify that Scheduler is LOCKED")
            locked = utils.check_scheduler_locked(self, sub_nr)
            if locked:
                Aqf.passed("Scheduler %s mode is 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s mode not 'locked'" % sub_nr)
            
            Aqf.hop("Verify that antennas are not pointing but still inhibited")
            Aqf.step("Verify that antennas are STOPPED")
            self._check_antennas_stopped()

            #Test cleanup
            Aqf.step("---Test cleanup---")
            #Resume operations
            Aqf.step("Resume operations")
            self.cam.sys.req.operator_resume_operations()
            Aqf.wait(3, "Wait for operations to resume")
            Aqf.step("Verify that Scheduler is ready (not locked)")
            ready = utils.check_scheduler_ready(self, sub_nr)
            if ready:
                Aqf.passed("Scheduler %s is ready - not mode 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % sub_nr)
            Aqf.step("Verify that antennas are ready to continue ")
            self._check_antennas_ready()

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))
            try:
                #Try to cleanup
                #Resume operations
                Aqf.step("Resume operations")
                self.cam.sys.req.operator_resume_operations()
            except:
                pass

        finally:
            teardown_subarray()

        Aqf.end()

    @aqf_vr("VR.CM.AUTO.I.1")
    def test_safety_control_operator_graceful_stop_observations(self):
        """Test actions when operator requests graceful stopping of observations."""

        def teardown_subarray():
            utils.teardown_subarray(self, 1)

        sub_nr = 1
        selected_ant = self._setup_subarray_and_verify(sub_nr)
        if not selected_ant:
            teardown_subarray()
            Aqf.end()
            return
        
        try: #Try the test
            # Create an OBSERVATION schedule block
            Aqf.step("Create two OBSERVATION schedule blocks")
            sb_id_code_1 = self._create_obs_sb(selected_ant, "VR.CM.AUTO.I.1")
            sb_id_code_2 = self._create_obs_sb(selected_ant, "VR.CM.AUTO.I.1")
            
            Aqf.step("Assign and schedule OBSERVATION SBs %s" % (sb_id_code_1))
            self.cam.subarray_1.req.assign_schedule_block(sb_id_code_1)
            self.cam.subarray_1.req.sb_schedule(sb_id_code_1)
            Aqf.step("Assign and schedule OBSERVATION SBs %s" % (sb_id_code_2))
            self.cam.subarray_1.req.assign_schedule_block(sb_id_code_2)
            self.cam.subarray_1.req.sb_schedule(sb_id_code_2)
            count = 30
            scheduled = False
            while (count > 0 and not scheduled):
                count -= 1
                observation_schedule = self.cam.sched.sensor.observation_schedule_1.get_value()
                scheduled = sb_id_code_1 in observation_schedule and sb_id_code_2 in observation_schedule
                Aqf.wait(2, "Waiting for SBs (%s and %s) to verify and be scheduled" % (sb_id_code_1,sb_id_code_2))
            if not scheduled:
                Aqf.failed("OBSERVATION SBs %s or %s was not scheduled. Exiting test." % (sb_id_code_1, sb_id_code_2))
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SBs %s and %s were scheduled.." % (sb_id_code_1, sb_id_code_2))

            Aqf.step("Execute OBSERVATION SB %s" % sb_id_code_1)
            self.cam.subarray_1.req.sb_execute(sb_id_code_1, timeout=10)
            count = 30
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

            # Allowing observation to run for few seconds before we graceful stop observations
            Aqf.step("Allow OBSERVATION SB %s to run for a few seconds "
                     "before operator intervenes to Stop Observations" % sb_id_code_1)
            Aqf.wait(15)

            Aqf.step("Inject operator request to Stop Observations")
            self.cam.sys.req.operator_stop_observations()
            Aqf.wait(3, "Wait for operator input to be processed")
            Aqf.step("Verifying that the necessary actions were successful")
            """
            Verification Requirements:
            --------------------------
            Test that the CAM implements the following
            upon receiving an intervention to Stop Observations:

            1. Complete currently executing SB
            2. Stop schedulers
            3. Verify that SBs cannot be executed until Resume Operations
            """
            Aqf.step("Verify that observations are stopped (Scheduler 1 mode is 'locked')")
            self._check_observations_stopped(1)
            Aqf.step("Verify that no observations are running on subarray 1")
            self._check_no_running_observations(1)
            Aqf.step("Verify that Scheduler is LOCKED")
            locked = utils.check_scheduler_locked(self, sub_nr)
            if locked:
                Aqf.passed("Scheduler %s mode is 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s mode not 'locked'" % sub_nr)
            

            Aqf.step("Verify that OBSERVATION SB %s cannot be executed" % sb_id_code_2)
            result = self.cam.subarray_1.req.sb_execute(sb_id_code_2, timeout=10)
            if result.succeeded:
                Aqf.failed("OBSERVATION SB %s should not have been activated. Exiting "
                           "test." % sb_id_code_2)
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SB %s could not be executed" % sb_id_code_2)

            Aqf.step("Resume Operations and verify that OBSERVATION SB %s can now be executed" % sb_id_code_2)
            Aqf.step("Inject operator request to Resume Operations")
            self.cam.sys.req.operator_resume_operations()
            Aqf.wait(3, "Wait for operations to resume")
            Aqf.step("Verify that Scheduler is ready (not locked)")
            ready = utils.check_scheduler_ready(self, sub_nr)
            if ready:
                Aqf.passed("Scheduler %s is ready - not mode 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % sub_nr)
                Aqf.end()
                return
            Aqf.step("Now verify that OBSERVATION SB %s can again be executed" % sb_id_code_2)
            result = self.cam.subarray_1.req.sb_execute(sb_id_code_2, timeout=10)
            if not result.succeeded:
                Aqf.failed("Request to execute OBSERVATION SB %s failed" % sb_id_code_2)
            else:
                Aqf.passed("Request to execute OBSERVATION SB %s succeeded" % sb_id_code_2)

            count = 30
            active = False
            while (count > 0 and not active):
                count -= 1
                active_schedule = self.cam.sched.sensor.active_schedule_1.get_value()
                active = sb_id_code_2 in active_schedule
                Aqf.wait(2, "Waiting for SB %s to start executing" % sb_id_code_2)
            if not active:
                Aqf.failed("OBSERVATION SB %s was not activated. Exiting "
                           "test." % sb_id_code_2)
            else:
                Aqf.passed("OBSERVATION SB %s is active as expected" % sb_id_code_2)

            #Test cleanup
            Aqf.step("---Test cleanup---")
            Aqf.hop("Waiting for SB to settle before stopping")
            Aqf.wait(15)
            Aqf.hop("Complete running SBs")
            result = self.cam.subarray_1.req.sb_complete(sb_id_code_2, True)
            Aqf.wait(4, "Wait for SB to complete")
            Aqf.step("Verify that Scheduler is ready (not locked)")
            ready = utils.check_scheduler_ready(self, sub_nr)
            if ready:
                Aqf.passed("Scheduler %s is ready - not mode 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % sub_nr)
                Aqf.end()
                return
            Aqf.step("Verify that no observations are running on subarray 1")
            self._check_no_running_observations(1)

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))

        finally:
            teardown_subarray()

        Aqf.end()


    @aqf_vr("CAM_script_quick_interrupt")
    def test_special_script_quick_interrupt(self):
        """Test actions when script is interrupted quickly."""

        def teardown_subarray():
            utils.teardown_subarray(self, 1)

        sub_nr = 1
        selected_ant = self._setup_subarray_and_verify(sub_nr)
        if not selected_ant:
            teardown_subarray()
            Aqf.end()
            return

        try:
            # Create an OBSERVATION schedule block
            Aqf.step("Create OBSERVATION schedule blocks")
            sb_id_code_1 = self._create_obs_sb(selected_ant, "CAM_script_quick_interrupt")
            
            Aqf.step("Assign and schedule OBSERVATION SBs %s" % (sb_id_code_1))
            self.cam.subarray_1.req.assign_schedule_block(sb_id_code_1)
            self.cam.subarray_1.req.sb_schedule(sb_id_code_1)
            count = 30
            scheduled = False
            while (count > 0 and not scheduled):
                count -= 1
                observation_schedule = self.cam.sched.sensor.observation_schedule_1.get_value()
                scheduled = sb_id_code_1 in observation_schedule
                Aqf.wait(2, "Wating for SBs (%s) to verify and be scheduled" % (sb_id_code_1))
            if not scheduled:
                Aqf.failed("OBSERVATION SBs %s was not scheduled. Exiting test." % (sb_id_code_1))
                Aqf.end()
                return
            else:
                Aqf.passed("OBSERVATION SBs %s was scheduled." % (sb_id_code_1))

            Aqf.step("Execute OBSERVATION SB %s" % sb_id_code_1)
            self.cam.subarray_1.req.sb_execute(sb_id_code_1, timeout=10)
            count = 30
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
                Aqf.passed("OBSERVATION SB %s is active as expected" \
                    % sb_id_code_1)

            # Allowing observation to run for few seconds before we graceful stop observations
            Aqf.step("Allow OBSERVATION SB %s to run for a while before stopping" % sb_id_code_1)
            Aqf.wait(6)
            self.cam.sys.req.operator_stop_observations()
            Aqf.wait(3, "Wait for operator input to be processed")
            Aqf.step("Verify that no observations are running on subarray 1")
            self._check_no_running_observations(sub_nr)

            Aqf.step("---Test cleanup---")
            Aqf.step("Wait for script kill to take effect - up to 30s")
            Aqf.sensor(self.cam.sched.sensor.active_schedule_1).wait_until('', sleep=1, counter=90)
            Aqf.hop("Resume scheduling")
            self.cam.sys.req.operator_resume_operations()
            Aqf.wait(3, "Wait for operations to resume")
            Aqf.step("Verify that Scheduler is ready (not locked)")
            ready = utils.check_scheduler_ready(self, sub_nr)
            if ready:
                Aqf.passed("Scheduler %s is ready - not mode 'locked'" % sub_nr)
            else:
                Aqf.failed("Scheduler %s is not ready - in mode 'locked'" % sub_nr)
            Aqf.passed("All ok")

        except Exception, err:
            tb = traceback.format_exc()
            Aqf.failed("EXCEPTION: %s \n %s" % (str(err), tb))

        finally:
            teardown_subarray()

        Aqf.end()


