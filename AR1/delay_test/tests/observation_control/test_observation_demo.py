from tests import settings, fixtures, specifics, Aqf, AqfTestCase
from nosekatreport import system, aqf_vr, site_acceptance, site_only
from katconf.sysconf import KatObsConfig
from katcorelib.katobslib.common import (ScheduleBlockStates,
                                         ScheduleBlockTypes,
                                         ScheduleBlockPriorities,
                                         ScheduleBlockVerificationStates)
from tests import (wait_sensor, wait_sensor_includes, wait_sensor_excludes, Aqf,
                   AqfTestCase, utils, specifics)
from katcorelib.katobslib.test import test_manager


def schedule_and_check_pass(aqfbase, db_manager, sub_nr,
                            schedule_block_id_code, timeout=30):
    """Test for success of scheduling a schedule block."""
    subarray = getattr(aqfbase.cam, 'subarray_{}'.format(sub_nr))
    schedule_sensor_name = 'observation_schedule_{}'.format(sub_nr)
    schedule_sensor_obj = getattr(aqfbase.cam.sched.sensor,
                                  schedule_sensor_name)
    reply = subarray.req.sb_schedule(schedule_block_id_code)
    Aqf.equals(reply.succeeded, True, 'Verify schedule request succeeded '
               'for schedule block %s.' % schedule_block_id_code)

    db_manager.expire()
    schedule_block = db_manager.get_schedule_block(
        schedule_block_id_code)
    if schedule_block.type == ScheduleBlockTypes.OBSERVATION:
        # Wait for verify to complete
        ok = wait_sensor_includes(
            aqfbase.cam, schedule_sensor_obj,
            schedule_block_id_code, timeout=timeout)
        Aqf.is_true(ok, 'cam.sched.sensor.{schedule_sensor_name} does not '
                    'contain {schedule_block_id_code}'.format(**locals()))
        db_manager.expire()
        schedule_block = db_manager.get_schedule_block(
            schedule_block_id_code)
        Aqf.equals(schedule_block.verification_state,
                   ScheduleBlockVerificationStates.VERIFIED,
                   'Verify schedule block %s is VERIFIED.' %
                   schedule_block_id_code)
    db_manager.expire()
    schedule_block = db_manager.get_schedule_block(
        schedule_block_id_code)
    Aqf.equals(schedule_block.state, ScheduleBlockStates.SCHEDULED,
               'Verify schedule block %s is SCHEDULED.' %
               schedule_block_id_code)


@system('all')
class TestObservationDemo(AqfTestCase):
    def setUp(self):
        # Free all subarrays and resources
        config = KatObsConfig(self.cam.system)
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        # Set the controlled resource
        self.controlled = specifics.get_controlled_data_proxy(Aqf)
        # Select an antenna and device to use
        self.ant_proxy = self.cam.ants[0].name
        self.extra_proxy = self.cam.ants[1].name
        self.subarrays = [sub for sub in self.cam.katpool.sensor.subarrays.get_value().split(",")]
        self.sub_objs = {}
        for sub_nr in self.subarrays:
            self.sub_objs[int(sub_nr)] = getattr(self.cam, "subarray_%s" % sub_nr)
        self.db_manager = test_manager.KatObsDbTestManager(config.db_uri)

    def tearDown(self):
        pass

    @aqf_vr('VR.CM.DEMO.OI.3')
    def test_graceful_stopping_observations(self):
        """Demonstrate CAM provides controls for the operators to stop observations and resume operations.
        """
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            sub_nr = 1
            subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
            Aqf.step("Open KatGUI and login as Lead Operator")
            Aqf.step("Goto SUBARRAYS.Observations")
            Aqf.step("Setting up a default subarray")
            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                antenna_spec=self.ant_proxy, controlled=self.controlled, activate=True)
            # Create an OBSERVATION schedule block
            Aqf.step("Creating an observation scheduling block.")
            sb_id1 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                    "CAM_basic_script", runtime=360, owner='VR.CM.DEMO.OI.3')
            Aqf.step("Created observation scheduling block: {}".format(sb_id1))

            Aqf.step('Waiting for subarray {} to be active'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=5)

            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                     "Verify scheduler is set to Manual Scheduling Mode")

            Aqf.step("Assigning and scheduling schedule block (%s) "
                      "to subarray %s." % (sb_id1, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id1)

            Aqf.step("Select and 'Execute' the OBSERVATION SB (%s) from the Observation Schedule table" % sb_id1)
            Aqf.checkbox("Verify that the newly created OBSERVATION SB is active")

            Aqf.step("Select 'Stop Observations' from the Operator Controls")
            Aqf.checkbox("Verify that the OBSERVATION SB (%s) was completed and moved to the Finished table" % sb_id1)
            Aqf.checkbox("Verify that the OBSERVATION SB (%s) has a STATE of INTERRUPTED" % sb_id1)
            #Aqf.checkbox("Verify that the OBSERVATION SB (%s) has an OUTCOME of SUCCESS or FAILURE)" % sb_id1)
            Aqf.checkbox("Verify that the Scheduler is in LOCKED state")
            Aqf.checkbox("Verify that the system Interlock state is OPERATOR_xxx_LOCK")

            Aqf.step("Select 'Resume Operations' from Operator Control Display")
            Aqf.checkbox("Verify that the Scheduler has exited LOCKED state and is ready for operations")
            Aqf.checkbox("Verify that the system Interlock state returned to NONE")
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBS.61')
    def test_demonstrate_operator_scheduling_inputs(self):
        """Demonstrate that CAM allows re-arrangement of sb's and execution of sb's."""
        sub_nr = 1
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.step("Open KatGUI and login as Control Authority")
            Aqf.step("Goto SUBARRAYS.SCHEDULE BLOCKS")
            Aqf.step("Creating 3 observation schedule blocks")
            sb_id1 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.61')
            sb_id2 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.61')
            sb_id3 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.61')
            Aqf.step("Created 3 schedule blocks: {}, {} and {}".format(sb_id1, sb_id2, sb_id3))

            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                    antenna_spec=self.ant_proxy, controlled=self.controlled, activate=False)
            #Aqf.step("Activate the subarray.")
            #Aqf.is_true(subarray.req.activate_subarray(timeout=90).succeeded,
            #            "Activation request for subarray {} successful".format(sub_nr))
            #Aqf.step('Waiting for subarray {} to be active.'.format(sub_nr))
            #ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=5)

            Aqf.step("Set scheduler to manual.")
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")

            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id1, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id1)
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id2, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id2)
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id3, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id3)

            sb_id = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.61')
            Aqf.step("Assigning schedule block (%s) "
                     "to subarray %s." % (sb_id, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id)

            Aqf.step("Login as Lead Operator")
            Aqf.step("Delegate SUBARRAY {} to a Control Authority".format(sub_nr))
            Aqf.step("In a 2nd KatGUI login as the Control Authority")
            Aqf.checkbox("Verify that you are logged in as Control Authority")
            Aqf.step("In the Lead operator KatGUI activate the subarray {}".format(sub_nr))
            Aqf.step("Re-arrange sb's by clicking the vertical ellipses that is in "
                     "the same row as sb {} and set priority to high.".format(sb_id))
            Aqf.checkbox("Verify that sb's are re-arranged and sb {} "
                         "is moved on top of the list.".format(sb_id))

            Aqf.step("Re-arrange sb's by clicking the vertical ellipses that is in "
                     "the same row as sb {} and set priority to low.".format(sb_id))
            Aqf.checkbox("Verify that sb's are re-arranged and sb {} "
                         "is no longer at the top of the list.".format(sb_id))

            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                    antenna_spec=self.ant_proxy, controlled=self.controlled, activate=False)

            Aqf.step("From KatGUI click 'SCHEDULER' then open 'SUBARRAY {} "
                     "OBSERVATION SCHEDULES'.".format(sub_nr))
            Aqf.step('Execute sb {} by clicking execute button'.format(sb_id3))
            Aqf.checkbox("Verify that the schedule block {} is executing.".format(sb_id3))
            Aqf.step('Stop execution of {} by clicking stop button'.format(sb_id3))
            Aqf.checkbox("Verify that the schedule block {} is stopped and "
                         "moved to 'Completed Schedule Blocks' table.".format(sb_id3))

            Aqf.checkbox("Using KatGUI clone sb {} from 'Completed Schedule Blocks' table to Draft.".format(sb_id3))
            Aqf.checkbox("Find the new schedule block and assign "
                         "to subarray %s and schedule SB" % (sub_nr))

            Aqf.step('Execute the newly cloned SB by clicking execute button')
            Aqf.checkbox("Verify that the schedule block has started.")
            Aqf.step('Stop execution of SB by clicking stop button')
            Aqf.checkbox("Verify that the schedule block is stopped")
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBR.56')
    def test_demonstrate_assign_schedule_blocks(self):
        """Demonstrate that cam assign schedule blocks"""
        sub_nr = 1
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.step("Open KatGUI and Login as Lead Operator")
            Aqf.step("Delegate SUBARRAY {} to a Control Authority".format(sub_nr))
            Aqf.step("In a 2nd KatGUI login as the Control Authority")
            Aqf.checkbox("Verify that you are logged in as Control Authority")

            Aqf.step("Goto SUBARRAYS.Schedule Blocks")
            sb_id = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBR.56')
            sb_id2 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBR.56')
            Aqf.hop("Created two schedule blocks for this test : {} and {}".format(sb_id, sb_id2))
            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                    antenna_spec=self.ant_proxy, controlled=self.controlled, activate=False)
            Aqf.hop("Created subarray {} for this test with {} and {}".format(sub_nr, ants_csv, controlled_csv))

            Aqf.step("Select schedule blocks (%s and %s) in Drafts and assign it to "
                     "subarray %s." % (sb_id, sb_id2, sub_nr))
            Aqf.checkbox("Verify that the two schedule blocks (%s and %s) "
                         "were assigned to subarray %s."% (sb_id, sb_id2, sub_nr))

            Aqf.step("Remove one of the schedule blocks from the subarray")
            Aqf.checkbox("Verify that the schedule block was removed from the subarray")

            Aqf.step("Assign the schedule block to the subarray again")
            Aqf.step("Schedule both schedule blocks ({} and {})".format(sb_id, sb_id2))

            Aqf.step("In KatGUI goto SUBARRAYS.Observations")
            Aqf.step("In the Lead operator KatGUI activate the subarray {}".format(sub_nr))
            Aqf.step('Execute sb {} by clicking execute button'.format(sb_id))
            Aqf.checkbox("Verify that the schedule block {} starts executing".format(sb_id))
            Aqf.step('Stop execution of {} by clicking stop button'.format(sb_id))
            Aqf.checkbox("Verify that the schedule block {} is Stopped and moved to "
                         "'Completed Schedule Blocks' table.".format(sb_id))

            Aqf.step('Start sb {} by clicking start button'.format(sb_id2))
            Aqf.checkbox("Verify that the schedule block {} starts executing".format(sb_id2))
            Aqf.step("Cancel execution of schedule block {} ".format(sb_id2))
            Aqf.checkbox("Verify that the schedule block {} is canceled and moved to "
                    "'Completed Schedule Blocks' table.".format(sb_id2))
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBS.19')
    def test_demonstrate_observation_schedule_display_components(self):
        """Demonstrate that cam observation schedule display components."""
        Aqf.step("Open KatGUI and log in as Lead Operator")
        Aqf.step("Goto SUBARRAYS.ALL OBSERVATIONS")
        Aqf.checkbox("Verify that there is a view for "
                     "simultaneous display of all subarrays")
        Aqf.checkbox("Verify that there is a view for "
                     "display per subarray")

        sub_nr = 1
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.hop("Creating a default subarray {}".format(sub_nr))
            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                    antenna_spec=self.ant_proxy, controlled=self.controlled, activate=False)
            Aqf.hop("Activating the subarray")
            Aqf.is_true(subarray.req.activate_subarray(timeout=15).succeeded,
                        "Activation request for subarray {} successful".format(sub_nr))
            Aqf.hop('Waiting for subarray {} to be active.'.format(sub_nr))
            ok = Aqf.sensor(subarray.sensor.state).wait_until("active", sleep=1, counter=5)

            Aqf.hop("Setting scheduler to manual")
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")

            Aqf.hop("Creating two new SBs")
            sb_id1 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.19')
            sb_id2 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.19')
            Aqf.hop("Assigning and scheduling schedule block (%s) "
                     "on subarray %s" % (sb_id1, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id1)
            Aqf.hop("Assigning and scheduling schedule block (%s) "
                     "on subarray %s" % (sb_id2, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id2)

            Aqf.step("Verify that SUBARRAY {} has list of scheduled SB - at least {} and {}".format(sub_nr, sb_id1, sb_id2))
            Aqf.checkbox("Verify that display indicates 'ready' when sufficient system resources "
                     "are available for an SB (e.g. resources not used by other SB and desired start time not in the future)")
            Aqf.step("Execute one of the schedule blocks-  {}".format(sb_id1))
            Aqf.checkbox("Verify that display now indicates 'not ready' for the other schedule block")
            Aqf.step("Stop the executing schedule block")
            Aqf.checkbox("Verify that display now indicates 'ready' for the other schedule block")
            Aqf.step("Click 'Free Subarray' button")
            Aqf.checkbox("Verify that all resources are removed from subarray {}".format(sub_nr))
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBS.60')
    def test_demonstrate_manual_queue_mode(self):
        """Demonstrate that cam provides manual and Queue mode."""
        sub_nr = 1
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.step("Open SCHEDULER display in KatGUI")
            sb_id1 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.60')
            sb_id2 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.60')
            sb_id3 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.60')
            Aqf.step("AQF created 3 schedule blocks {}, {} and {}".format(sb_id1, sb_id2, sb_id3))
            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                    antenna_spec=self.ant_proxy, controlled=self.controlled, activate=False)

            Aqf.step("Setting scheduler to manual")
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id1, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id1)
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id2, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id2)
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id3, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id3)

            #Aqf.step("Open SCHEDULER display in KatGUI")
            Aqf.step("Login as Lead Operator")
            Aqf.step("Delegate SUBARRAY {} to a Control Authority".format(sub_nr))
            Aqf.step("In a 2nd KatGUI login as the Control Authority")
            Aqf.checkbox("Verify that you are logged in as Control Authority")
            Aqf.step("In the Lead operator KatGUI activate the subarray {}".format(sub_nr))

            Aqf.step("Change SUBARRAY 1 mode to queue")
            Aqf.checkbox("Verify that sb's start executing automatically according to "
                         "the way they are ordered in queue mode (wait for at least 2 SBs to execute)")

            Aqf.step("Change mode to manual")
            Aqf.checkbox("Verify that mode was changed to manual")
            Aqf.step("Creating two more schedule blocks")
            sb_id4 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.60')

            sb_id5 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                        "CAM_basic_script", runtime=10, owner='VR.CM.DEMO.OBS.60')
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id4, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id4)
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s." % (sb_id5, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id5)

            Aqf.checkbox("Wait a while and verify that sb's are not started automatically "
                         "when in manual mode")
            Aqf.step("Execute sb {}.".format(sb_id4))
            Aqf.checkbox("Verify that sb {} has been manually executed".format(sb_id4))
            Aqf.checkbox("Verify that sb {} does not start automatically on completion of the executing SB".format(sb_id5))

            Aqf.step("Stop any executing SBs on subarray {}".format(sub_nr))
            Aqf.step("Unassign any remaining SBs from the Observation Schedule {}".format(sub_nr))
            Aqf.step("Free subarray {}".format(sub_nr))
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBR.64')
    def test_demonstrate_add_resource_to_sub_array(self):
        """Demonstrate that CAM add resouce to subarray."""
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.step("Open SUBARRAY.RESOURCES display in KatGUI")
            Aqf.step("From the Free Resources, select two resources and add them to a subarray")
            Aqf.checkbox("Verify that resources were moved to the subarray")

            Aqf.step("Remove one of the resources from the subarray")
            Aqf.checkbox("Verify that the resource was moved from the subarray to the Free Resources")

            Aqf.step("Activate the subarray")
            Aqf.checkbox("Verify that the resources cannot be added to or removed from the subarray while active")

            Aqf.step("Free the subarray")
            Aqf.checkbox("Verify that all resources were removed from the subarray")
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBR.65')
    def test_demonstrate_managing_resource_pools(self):
        """Demonstrate that CAM is managing resource pools"""
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.step("From KatGUI select 'SET-UP SUBS' then 'SET-UP SUBARRAY'")

            Aqf.checkbox("Verify that free resources are displayed in the resources table")

            Aqf.step("In the Resources panel, select resource")
            Aqf.step("Add the selected resource to preferred subarray 1")
            Aqf.checkbox("Verify that the resource has moved to subarray 1")

            Aqf.step("In the Resources panel, mark another resource as in-maintanace")
            Aqf.checkbox("Verify that the resource marked as in-maintanance change colour"
                         " and a maintenance icon (spanner) is displayed next to the resource")

            Aqf.checkbox("Verify that adding the in-maintenance resource to the subarray 1 fails")
            Aqf.step("Set the maintenance flag of subarray 1")
            Aqf.step("Add the in-maintenance resource to the maintenance subarray 1")
            Aqf.checkbox("Verify that the resource has now moved to subarray 1")

            Aqf.step("In the Subarray panel, remove the in-maintenance resource from subarray")
            Aqf.step("In the Resources panel, take resource out of maintenance")
            Aqf.checkbox("Verify that the resource is taken out of maintenance by changing colour"
                         " and that the maintenance icon is removed ")

            Aqf.step("Click 'Free Subarray' button")
            Aqf.checkbox("Verify that the resource is removed from subarray 1 "
                         "and are now available in the Resources panel "
                         "of the 'RESOURCES' display")
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")

        Aqf.end()


    @aqf_vr("VR.CM.DEMO.OBR.69")
    def test_limit_resource_control(self):
        """Demonstrate the CAM limits resource control to those resources delegated to them."""

        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.step("In KatGUI open OBSERVATION display")
            sub_nr = 1
            ant1 = self.cam.ants[0].name
            ant2 = self.cam.ants[1].name
            ants = ",".join([ant1,ant2])
            controlled = specifics.get_controlled_data_proxy(self, sub_nr=sub_nr)
            res_csv = ",".join([ant1, ant2, controlled])
            Aqf.step("The system will now setup and activate subarray_1 with resources {}".format(res_csv))
            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                    antenna_spec=ants, controlled=self.controlled, activate=True)
            Aqf.step("The system will now create and execute a manual SB for {}".format(ant1))
            sb_id = utils.create_manual_sb(self, owner="VR.CM.DEMO.OBR.69", antenna_spec=ant1, controlled=controlled_csv)
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s" % (sb_id, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id)
            Aqf.step("Execute manual SB {} on subarray {}".format(sb_id, sub_nr))
            Aqf.checkbox("Verify that manual SB is {} executing on subarray {}".format(sb_id, sub_nr))
            Aqf.step("Open iPython on the OBS node of your test site")
            Aqf.step("Enter: import katuilib; configure_subarray(sub_nr='{}')".format(sub_nr))
            Aqf.step("Enter: kat.connected_objects")
            Aqf.checkbox("Verify that only the resources assigned to the subarray "
                "are presented in the kat container (i.e. connected_objects), including {}".format(res_csv))
            Aqf.step("Enter: kat.controlled_objects")
            Aqf.checkbox("Verify that none of the resources are in the controlled_objects")
            Aqf.step("Enter: kat.{}.req.<tab>".format(ant1))
            Aqf.checkbox("Verify that {} does not expose the requests to control the receptor".format(ant1))
            Aqf.step("Enter: configure_sb('{}')".format(sb_id))
            Aqf.step("Enter: kat.controlled_objects")
            Aqf.checkbox("Verify that only the resources allocated to the manual SB "
                "are in the controlled_objects: namely {} and {} and not {}".format(ant1, controlled, ant2))
            Aqf.step("Enter: kat.{}.req.<tab>".format(ant1))
            Aqf.checkbox("Verify that {} exposes the requests to control the receptor".format(ant1))
            Aqf.step("From KatGUI, attempt to allocate resources in subarray {} to another subarray".format(sub_nr))
            Aqf.checkbox("Verify that assigned resources '{}' cannot be assigned to another subarray".format(res_csv))
            Aqf.step("From KatGUI, stop the manual SB {} on subarray {}".format(sb_id, sub_nr))
            Aqf.step("From KatGUI, free subarray {}".format(sub_nr))
            Aqf.checkbox("Verify that all resources from subarray {} was freed".format(sub_nr))
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBRR.44')
    def test_manual_control_demo(self):
        """Demonstrate CAM command line control functionality for expert user.
        """
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        try:
            Aqf.step("In KatGUI open OBSERVATION display")
            sub_nr = 1
            ant1 = self.cam.ants[0].name
            ant2 = self.cam.ants[1].name
            ants = ",".join([ant1,ant2])
            controlled = specifics.get_controlled_data_proxy(self, sub_nr=sub_nr)
            res_csv = ",".join([ant1, ant2, controlled])
            Aqf.step("The system will now setup and activate subarray_1 with resources {}".format(res_csv))
            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                    antenna_spec=self.ant_proxy, controlled=self.controlled, activate=True)
            Aqf.step("The system will now create and execute a manual SB for {}".format(ant1))
            sb_id = utils.create_manual_sb(self, owner="VR.CM.DEMO.OBRR.44", antenna_spec=ant1, controlled=controlled_csv)
            Aqf.step("Assigning schedule block (%s) "
                         "to subarray %s" % (sb_id, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id)
            Aqf.step("Execute manual SB {} on subarray {}".format(sb_id, sub_nr))
            Aqf.checkbox("Verify that manual SB is {} executing on subarray {}".format(sb_id, sub_nr))

            Aqf.step("Open an iPython session do the following:")
            Aqf.step("import katuilib; configure_sb('{}');".format(sb_id))
            Aqf.step("Verify that manual control of {} is exposed through the kat object".format(ant1))
            Aqf.step("Execute manual control of {}".format(ant1))
            Aqf.step("e.g. use kat.{}.req.target_azel(80,80)".format(ant1))
            Aqf.step("and kat.{}.req.mode('POINT')".format(ant1))
            Aqf.step("and kat.print_sensors('pos.actual', '{}', 'period', 1)".format(ant1))
            Aqf.step("and use kat.{}.req.mode('STOP')".format(ant1))
            Aqf.checkbox("Confirm manual control of the subarray's resources were possible")

            Aqf.step("From KatGUI, stop the manual SB {} on subarray {}".format(sb_id, sub_nr))
            Aqf.step("From KatGUI, free subarray {}".format(sub_nr))
            Aqf.checkbox("Verify that all resources from subarray {} was freed".format(sub_nr))
        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")

        Aqf.end()


@system('all')
class TestMetadataDemo(AqfTestCase):

    @aqf_vr('VR.CM.DEMO.C.40')
    def test_demonstrate_cam_expose_cam2spead(self):
        """Demonstrate that CAM expose cam2spead device"""
        Aqf.step("Login to KatGUI and open Sensor List")
        Aqf.step("Click DATA_1, DATA_2, DATA_3 or DATA_4")
        Aqf.checkbox("Verify that in all data proxies there is "
                     "a cam2spead device with sensors e.g cam2spead-connected")
        Aqf.step("Open Instrumental Configuration on KatGUI and view user/cam2spead/mkat.conf")
        Aqf.checkbox("Verify that meta data sensors for delivery by cam2spead are configured")
        Aqf.end()

    @site_only
    @aqf_vr('VR.CM.SITE.C.25')
    def test_demonstrate_meta_data_site(self):
        """Demonstrate that CAM provides meta-data augment data products"""
        Aqf.step("Run an OBSERVATION schedule block")
        Aqf.checkbox("Verify that configured metadata is saved in the HDF5 data product file")
        Aqf.end()


