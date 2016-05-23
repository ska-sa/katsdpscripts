###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time

from nosekatreport import (aqf_vr, system, site_acceptance, site_only)
from tests import (utils, settings, specifics, Aqf, AqfTestCase)
from tests.observation_control.test_observation_demo import schedule_and_check_pass
from katconf.sysconf import KatObsConfig
from katcorelib.katobslib.test import test_manager
from katcorelib.katobslib.common import ScheduleBlockTypes

@system('all')
class TestOperations(AqfTestCase):

    """Tests for operations."""

    def setUp(self):
        pass

    def tearDown(self):
        self.cam.subarray_1.req.free_subarray()
        pass

    def wait_sensor_value(self, katsensor, look_for, timeout):
        """Test katsensor value every second to see if it contains a string, until timeout"""
        i = 0
        while i < timeout:
            val = katsensor.get_value()
            if look_for in str(val):
                return True
            i = i + 1
            time.sleep(1.0)
        return False

    @aqf_vr('VR.CM.DEMO.DS.9')
    def test_health_display(self):
        """Demonstrate the overall system health display.
        """
        Aqf.step("Open the KATGUI Health display")
        Aqf.checkbox("Verify the overall system health is presented")

        specifics.simulate_intrusion(self)

        Aqf.checkbox("Verify on KATGUI Health Display that Intrusion OK sensor "
                    "is not nominal")
        Aqf.checkbox("Verify on KATGUI Health Display that ANC is not nominal")

        specifics.reset_intrusion(self)

        Aqf.checkbox("Verify on KATGUI that Intrusion OK returns to nominal")
        Aqf.checkbox("Verify on KATGUI that ANC returns to nominal")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DS.1')
    def test_display_serial_numbers(self):
        """Demonstrate that CAM displays serial numbers.
        """
        Aqf.step("Open the KATGUI Device Status Display")
        Aqf.checkbox("Verify that serial number sensors are displayed")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DS.2')
    def test_firmware_versions_info(self):
        """Demonstrate that CAM provides a display for software and firmware versions.
        """
        Aqf.step("Open the KATGUI Sensor List Display")
        Aqf.step("Select ANC from the component panel")

        Aqf.step("Search build and version sensors by entering 'version' in the search box")
        Aqf.checkbox("Verify that version sensors include that of "
                     "the devices(vds, tfr, pdu) managed by ANC")

    @aqf_vr('VR.CM.DEMO.DS.14')
    def test_build_info(self):
        """Demonstrate that CAM provides a display for system build state.
        """
        Aqf.step("Open the KATGUI Sensor List Display")
        Aqf.step("Select ANC from the component panel")

        Aqf.step("Search build sensors by entering 'build' in the search box")
        Aqf.checkbox("Verify that build sensors include that of "
                     "the devices(vds, tfr, pdu) managed by ANC")
        Aqf.step("Select 'EXE' from the component panel")
        Aqf.checkbox("Verify that build sensors are displayed")

        # Aqf.step("Open the KATGUI Sensor Snapshot Display")
        # Aqf.step("Select any one of the Proxies from the Node tree")
        # Aqf.step("Select the Build filter button")
        #Aqf.checkbox("Verify the build and version sensors on the selected proxy are displayed")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.MSG.11')
    def test_display_values_of_selected_monitoring_points(self):
        """Test that the GUI display values of selected monitoring points
            Steps:
            ------
            1. Verify that interface allows selection of one or more monitoring points
               for display.
            2. Verify that CAM display shows the values of selected monitoring points.
            3. Verify the displayed monitoring points are correct (as expected).
        """
        Aqf.step("Select 'Sensor List display' from KATGUI")
        Aqf.step("Select any receptor/antenna on the node tree")
        Aqf.checkbox("Verify that a snapshot of the selected antenna/receptor sensors is displayed")
        Aqf.step("Filter on '-azim'")
        Aqf.checkbox("Verify that all '-azim' sensors are filtered and updated periodically")

        Aqf.step("Click on the ap.actual-azim sensor")
        Aqf.checkbox("Verify that the 'ap.actual-azim' sensor for receptor M011 is plotted")

        azim = 100
        Aqf.step("Point antenna to change the request-azim (and thus actual-azim) value to %s" % azim)
        self.cam.ants.req.target_azel(azim,45)
        self.cam.ants.req.mode("POINT")
        Aqf.checkbox("Verify that the request-azim value is %s" % azim)
        Aqf.checkbox("Verify that the actual-azim value settles to %s on the plot" % azim)

        azim = 120
        Aqf.step("Change request-azim position to %s" % azim)
        self.cam.ants.req.target_azel(azim,60)
        self.cam.ants.req.mode("POINT")
        Aqf.checkbox("Verify that the request-azim value is %s" % azim)
        Aqf.checkbox("Verify that the actual-azim value settles to %s on the plot" % azim)

        Aqf.step("Stop the receptor/antenna")
        self.cam.ants.req.mode("STOP")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DS.30')
    def test_health_display_support_fault_finding(self):
        """Demonstrate that CAM health display supports fault finding.
        """

        Aqf.step("Open the KATGUI Top Health display")
        Aqf.step("Hover over a node, or in the case of an agg_* sensor click the node")
        Aqf.step("Verify that the rule, status and values of the sensors are shown")
        Aqf.checkbox("Verify that information is sufficient to identify the subsystem/origin of the fault")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DS.30')
    def test_displays_support_fault_finding_per_subsystem(self):
        """Demonstrate CAM displays facilitate fault finding per subsystem
        """
        Aqf.step("From KatGUI open 'SENSOR LIST DISPLAY'")
        Aqf.step("Click on one of the Proxies")
        Aqf.step("Select the checkbox to 'Hide Nominal'")
        Aqf.checkbox("Verify that all sensors with status != nominal are displayed")
        Aqf.step("Deselect the checkbox to 'Hide Nominal'")
        Aqf.checkbox("Verify that all sensors, including those with status == nominal, are displayed")

        Aqf.step("Click on one of the Receptors")
        Aqf.step("Search on 'dig' or 'rsc'")
        Aqf.step("Click on Hide Nominal")
        Aqf.checkbox("Verify that this display can be used to determine the failures on a specific subsystem")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DP.25')
    def test_system_status_displays_horizon_mask_demo(self):
        """Demonstrate CAM provides system status displays for Horizon masks.
        """

        Aqf.step("Verify that Horizon Masks can be displayed")
        Aqf.step("Open KATGUI Pointing Display")
        Aqf.step("Select one or more antennas/receptors to display the Horizon Mask")
        Aqf.step("NOTE: the test configuration has dummy horizon mask files for some but not all antennas/receptors")
        Aqf.step("NOTE: if necessary a horizon mask for a specific receptor can be copied to static/horizon-masks/ ")
        Aqf.step("Verify that a horizon masks is displayed for some receptors")
        Aqf.step("Verify that a message is displayed if a receptor does not have an horizon mask")
        Aqf.checkbox("Horizon masks are displayed as expected")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DP.25')
    def test_system_status_displays_additional_settings_demo(self):
        """
        Demonstrate CAM provides additional settings in per-subarray display as required:
        Receiver settings, Digitiser settings, Correlator settings
        """

        Aqf.step("Open KatGUI Subarray Resources Display")
        Aqf.step("Configure the subarray with resources, band and product, and Control Authority")
        Aqf.step("Verify that the Selected Receiver band is displayed")
        Aqf.step("Verify that the Selected Product is displayed")
        Aqf.step("Verify that the Control Authority is displayed")
        Aqf.step("Additional settings for Receiver, Digitiser, Correlator per subarray is not required for now (Lindsey)")
        Aqf.checkbox("The subarray configuration was displayed")
        Aqf.step("Free the subarray")
        Aqf.checkbox("Verify that the Selected Receiver band, Product and Control Authority have been reset")

        Aqf.end()

    def _create_and_schedule_obs_sb(self, ant_id, runtime=30, program_block="AQF"):
        sub_nr = 1
        Aqf.step("Creating an OBSERVATION SB for receptor/antenna %s" % ant_id)
        controlled_resources=specifics.get_controlled_data_proxy(self)
        sb_id_code = self.obs.sb.new(owner='aqf_test', antenna_spec=ant_id,
                controlled_resources=controlled_resources)
        self.obs.sb.description = "Point source scan for %s" % ant_id
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        # quick scan over the sun
        self.obs.sb.instruction_set = (
            "run-obs-script ~/svn/katscripts/cam/basic-script.py -t 3 -m {runtime} "
            "--proposal-id=CAM_AQF --program-block-id={program_block}".
            format(**locals()))
        self.obs.sb.unload()
        res_csv = ",".join([ant_id,controlled_resources])
        Aqf.step("Assigning %s to subarray_1" % res_csv)
        self.cam.subarray_1.req.assign_resources(res_csv)
        Aqf.step("Assigning %s to subarray_1" % sb_id_code)
        self.cam.subarray_1.req.assign_schedule_block(sb_id_code)
        Aqf.step("Activate to subarray_1")
        self.cam.subarray_1.req.activate_subarray(timeout=15)
        Aqf.step("Set scheduler 1 to manual")
        self.cam.sched.req.mode(1, 'manual')
        Aqf.step("Scheduling %s on subarray_1" % sb_id_code)
        self.cam.sched.req.sb_schedule(1, sb_id_code)
        Aqf.step("Wait for SB (%s) to complete verifying" % sb_id_code)
        ok = self.wait_sensor_value(self.cam.sched.sensor.observation_schedule_1, sb_id_code, 180)
        Aqf.checkbox("Verify the newly created OBSERVATION SB (%s) for %s "
            "in the Observation Schedule table" % (sb_id_code, ant_id))
        return sb_id_code

    @site_acceptance
    @aqf_vr('VR.CM.DEMO.OI.4')
    def test_cam_safety_controls_operator_stow(self):
        """Demonstrate CAM provides safety controls for the operators on KatGUI -
        operator stow antennas
        """
        # Operator to stow all antennas.
        Aqf.step("Open the KATGUI Scheduler Display")
        ant_id = self.cam.ants[0].name
        sb_id_code = self._create_and_schedule_obs_sb(ant_id, program_block="VR.CM.DEMO.OI.4")

        Aqf.step("Select and 'Execute' the OBSERVATION SB (%s) from the Observation Schedule table" % sb_id_code)
        Aqf.checkbox("Verify that the newly created OBSERVATION SB is executing")

        Aqf.step("Open the Control Display and click 'Operator Control'")
        Aqf.step("Select 'Stow All' from the KATGUI")
        Aqf.checkbox("Verify the antennas are STOW on the Operator Control Display")
        Aqf.checkbox("Verify the antennas are STOW on the Pointing Display")

        Aqf.step("Select 'Resume Operations' from Operator Control Display")
        Aqf.checkbox("Verify the antennas are STOP on the Operator Control Display")
        Aqf.checkbox("Verify the antennas are STOP on the Pointing Display")

        #Cleanup - if required
        self.cam.subarray_1.req.sb_to_draft(sb_id_code)
        Aqf.end()

    @site_acceptance
    @aqf_vr('VR.CM.DEMO.OI.4')
    def test_cam_safety_controls_operator_inhibit(self):
        """Demonstrate CAM provides safety controls for the operators on KatGUI -
        operator inhibit antennas.
        """
        # Operator to prevent movement of antennas (inhibit)
        Aqf.step("Open the KATGUI Control display - Operator control")
        ant_id = self.cam.ants[0].name
        sb_id_code = self._create_and_schedule_obs_sb(ant_id, program_block="VR.CM.DEMO.OI.4")

        Aqf.step("Select and 'Execute' the OBSERVATION SB (%s) from the Observation Schedule table" % sb_id_code)
        Aqf.checkbox("Verify that the newly created OBSERVATION SB is executing")

        Aqf.step("Select 'Inhibit All' from the KATGUI")
        #Aqf.checkbox("Verify the antennas are inhibited and stops moving immediately without stowing")
        Aqf.checkbox("Verify the antennas are STOP and INHIBIT on the Operator Control Display")
        Aqf.checkbox("Verify the antennas are shown as INHIBIT on the Pointing Display")
        Aqf.step("Open the Scheduler Display")
        Aqf.checkbox("Verify that the OBSERVATION SB (%s) was interrupted and moved to the Completed table" % sb_id_code)
        Aqf.checkbox("Verify that the Schedulers are in LOCKED mode")

        Aqf.step("Select 'Resume Operations' from Operator Control Display")
        Aqf.checkbox("Verify the antennas are STOP and NOT inhibited on the Operator Control Display")
        Aqf.checkbox("Verify the antennas are NOT inhibited on the Pointing Display")
        Aqf.checkbox("Verify that the Schedulers are no longer in LOCKED mode")

        #Cleanup - if required
        self.cam.subarray_1.req.sb_to_draft(sb_id_code)

        # Operator to gracefully power down sensitive equipment
        # This test is in zzLast

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OBR.6')
    def test_cam_reset_and_restart_components(self):
        """Demonstrate CAM provides an interface to restart components in maintenance remotely.
        """
        Aqf.step("Login to KATGUI as Lead Operator")
        Aqf.step("Open activity log for verification. 'Home'->'System Logs'->'activity.log'")
        Aqf.step("In a 2nd window, goto SUABARRAY.Resources")
        Aqf.step("Set the maintenance flag on one of the subarrays")
        Aqf.step("Set one of the receptor proxies to in-maintenance in the Resources panel")
        Aqf.step("Assign the in-maintenance resource to the maintenance subarray")
        Aqf.step("Click on the Restart Maintenance Device button of the receptor proxy")
        Aqf.step("In the popup select the 'AP' device to restart")
        Aqf.checkbox("Verify that 'AP' was restarted by checking the activity log")

        Aqf.keywait("Close the activity log window")
        Aqf.keywait("Free the subarray and take the receptor proxy and subarray out of maintenance")

        Aqf.end()

    @aqf_vr("VR.CM.DEMO.CA.34")
    def test_inspect_structural_pointing_corrections_applied(self):
        """Inspect the CAM code to insure that the Structural Pointing Models are applied."""
        filename = "svn/katproxy/katproxy/proxy/mkat_receptor_proxy.py"
        #apply structural pointing corrections
        search = "pointing.apply and pointing.reverse"
        Aqf.step("Do a code inspection to insure that structural pointing "
                 "corrections are applied")
        Aqf.checkbox("Do a code inspection on %s: search for %s" %
                     (filename, search))
        Aqf.end()

    @aqf_vr("VR.CM.DEMO.CA.35")
    def test_inspect_thermal_pointing_corrections_applied(self):
        """Inspect the CAM code to insure that the Thermal Pointing Models are applied."""
        filename = "svn/katproxy/katproxy/proxy/mkat_receptor_proxy.py"
        #apply thermal pointing corrections
        search = "thermal.apply and thermal.reverse"
        Aqf.step("Do a code inspection to insure that thermal pointing "
                 "corrections are applied")
        Aqf.step("Thermal model not yet implemented.")
        Aqf.waived(" Waiting for tilt temperature tests to determine way forward")
        Aqf.end()

    @aqf_vr("VR.CM.DEMO.CA.36")
    def test_inspect_refraction_pointing_corrections_applied(self):
        """Inspect the CAM code to insure that the Pointing Models are used."""
        filename = "svn/katproxy/katproxy/proxy/mkat_receptor_proxy.py"
        #apply refraction pointing corrections
        search = "refraction.apply and refraction.reverse"
        Aqf.step("Do a code inspection to insure that refraction pointing "
                 "corrections are applied")
        Aqf.checkbox("Do a code inspection on %s: search for %s" %
                     (filename, search))
        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.DEMO.CFG.37")
    def test_display_catalogues(self):
        """Demonstrate that the KATGUI displays the source catalogues
        """
        Aqf.step("Open the KATGUI Instrumental configuration display")
        Aqf.step("Select 'source_list.csv' under Catalogues panel")
        Aqf.checkbox("Verify that the sources are displayed")

        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.DEMO.V.46")
    def test_the_display_of_a_receptor_number(self):
        """Demonstrate that the receptor number for which the VDS video feed should be displayed"""
        Aqf.step("Open the KatGui Video Display.")
        Aqf.step("Pan camera to 70.")
        Aqf.step("Tilt camera to 20.")
        Aqf.step("Set preset by clicking 'Set Preset'.")
        Aqf.step("Select the preset receptor nr (e.g. m011) to set.")
        Aqf.step("Submit the preset by clicking 'SET SELECTED PRESET' button.")

        Aqf.step("Pan camera to 10.")
        Aqf.step("Tilt camera to 10.")
        Aqf.step("Go to preset by clicking 'Go to Preset' button.")
        Aqf.step("Select the previously set receptor nr (e.g. m011) to goto.")
        Aqf.checkbox("Verify that the video sensors return to the selected preset"
                     " position {pan: 70, tilt: 20}.")
        Aqf.checkbox("On site - verify that the receptor number is displayed on the VDS video feed.")

        Aqf.end()


    @aqf_vr("VR.CM.SITE.V.26")
    def test_cam_display_video(self):
        """Demonstrate that the CAM provides a view for displaying the video feed from the Camera"""
        Aqf.step("Open the KatGUI Display.")
        Aqf.step("Click Video display.")
        Aqf.checkbox("Verify that the video feed from the camera is displayed")

        Aqf.end()

    @site_acceptance
    @aqf_vr("VR.CM.DEMO.V.17")
    def test_video_display_subsystem_control(self):
        """Demonstrate that the GUI provides Video Display Subsystem Control."""
        Aqf.step("From KatGUI open Sensor List display, click on ANC and filter on VDS")
        Aqf.step("From KatGUI open Video display")
        Aqf.step("Pan camera to 70.")
        Aqf.step("Tilt camera to 20.")
        Aqf.step("Set preset by clicking 'Set Preset'.")
        Aqf.step("Select the preset receptor nr (e.g. m001) to set.")
        Aqf.step("Submit the preset by clicking 'SET SELECTED PRESET' button.")
        Aqf.step("Pan camera far left")
        Aqf.checkbox("Verify that camera is moving to the left side and the pan sensor value changes")
        Aqf.step("Zoom camera wide")
        Aqf.checkbox("Verify that camera is zooming wide and the zoom sensor value changes")
        Aqf.step("Go to preset by clicking 'Go to Preset' button and choose receptor nr 'm001'.")
        Aqf.checkbox("Verify that the video sensors return to the selected preset"
                     " position {pan: 70, tilt: 20}.")

        Aqf.end()

    @aqf_vr('VR.CM.SITE.OP.3')
    def test_demonstrate_control_centre_losberg(self):
        """Demonstrate that CAM supports a control centre in Losberg."""

        Aqf.step("Open KatGUI in Losberg")
        Aqf.step("Login as 'lead operator'")
        Aqf.checkbox("Verify that you are logged in.")

        Aqf.end()

    @aqf_vr('VR.CM.SITE.OP.4')
    def test_demonstrate_single_lead_operator(self):
        """Demonstrate that CAM allows only one lead operator at any time to be logged in."""

        Aqf.step("Open two KatGUIs")
        Aqf.step("Login as 'lead operator' on first KatGUI")
        Aqf.checkbox("Verify that identification and authorisation was required to login")
        Aqf.checkbox("Verify that the current Lead Operator is shown on KatGUI")

        Aqf.step("Login as 'lead operator' on the second KatGUI")
        Aqf.step("Verify that the second KatGUI operator is the new lead operator")
        Aqf.step("Verify that the Lead Operator on the first KatGUI is now logged out")
        Aqf.step("Verify that the Lead Operator can delegate control of a specific subarray to a Control Authority")
        Aqf.step("Verify that the Lead Operator can revoke control of a specific subarray from the Control Authority through assigning control back to LO")
        Aqf.checkbox("A single lead oeprator is enforced")

        Aqf.step("Logout as 'lead operator' on the second KatGUI")
        Aqf.step("Login  as a new 'lead operator' on the second KatGUI")
        Aqf.step("Verify that the new lead operator is now registered")
        Aqf.checkbox("Lead operator transfer is allowed")

        Aqf.step("At a different location login as 'lead operator' on a 3rd KatGUI")
        Aqf.step("Verify that the new lead operator is now registered")
        Aqf.checkbox("Lead operator transfer at different control centres is allowed")

        Aqf.end()

    @aqf_vr('VR.CM.SITE.Z.6')
    def test_demonstrate_server_two_racks(self):
        """Demonstrate that cam servers are limited to two racks."""
        Aqf.checkbox("Verify that cam servers are limited to two racks in KAPB.")
        Aqf.end()

    @aqf_vr('VR.CM.SITE.Z.7')
    def test_demonstrate_costs_equipment(self):
        """Inspect that cam use COSTS equipment."""
        Aqf.step("Pull server out of the rack.")
        Aqf.checkbox("Verify that cam servers from Dell.")
        Aqf.checkbox("Verify that cam servers have serial numbers.")
        Aqf.end()

    @aqf_vr('VR.CM.SITE.A.10')
    def test_demonstrate_fire_alarm_kapb(self):
        """Inspect that cam fire alarm is raised at site."""
        Aqf.step("Open 'HEALTH AND STATE' window in KatGUI")

        Aqf.step("Login to the BMS as the Engineer")
        Aqf.step("Open CAM Simulation screen in BMS UI.")
        Aqf.step("Trigger fire by toggling fire button to 'on'.")

        Aqf.checkbox("Verify that the Health display indicates the 'fire ok' as an error")
        Aqf.step("Open 'Alarms' window on KatGUI")
        Aqf.checkbox("Verify that the System_Fire alarm is indicated as a critical alarm on the Alarms display")

        Aqf.step("Open CAM Simulation screen in BMS UI.")
        Aqf.step("Trigger fire by toggling fire button to 'off'.")

        Aqf.checkbox("Verify that the System_Fire alarm returns to nominal on the Alarms display")
        Aqf.step("Clear the System_Fire nominal alarm on the Alarms display")
        Aqf.checkbox("Verify that the System_Fire alarm is nominal and cleared on the Alarms display")

        Aqf.end()

    @aqf_vr('VR.CM.SITE.A.11')
    def test_demonstrate_site_cooling_alarm_kapb(self):
        """Inspect that cam cooling alarm is raised at site."""
        Aqf.step("Login to the BMS as the Engineer")
        Aqf.step("Open CAM Simulation screen in BMS UI.")

        Aqf.step("NOTE: If your verification takes longer than the alarm delay "
                 "(60s) the computing will be shutdown !!!")
        Aqf.checkbox("OK to continue?")

        Aqf.step("Trigger cooling by toggling cooling button to 'on'.")

        Aqf.checkbox("Verify on the Health display that the 'cooling ok' indicator "
                     "has been set to an error value")
        Aqf.checkbox("Verify on the Alarms display that the System_Cooling_Failure "
                     "alarm is indicated as a critical alarm")

        Aqf.checkbox("Verify on the Health display that 'cooling ok' indicator to normal")
        Aqf.checkbox("Verify on the Alarms display that the System_Cooling_Failure "
                     "alarm returns to nominal")

        Aqf.step("Clear the System_Cooling_Failure nominal alarm on the Alarms display")
        Aqf.checkbox("Verify that the System_Cooling_Failure alarm is nominal and "
                     "cleared on the Alarms display")

        Aqf.end()

    @aqf_vr('VR.CM.SITE.A.12')
    def test_demonstrate_site_temperature_alarm_kapb(self):
        """Inspect that cam temperature alarm is raised at site."""
        Aqf.step("Login to the BMS as the Engineer")
        Aqf.step("Open CAM Simulation screen in BMS UI.")
        Aqf.step("Set temperature value to 30 and submit.")

        Aqf.checkbox("Verify on the Health display that the 'temperature ok' indicator "
                     "has been set to an error value")
        Aqf.checkbox("Verify on the Alarms display that the System_Temperature_Failure "
                     "alarm is indicated as a critical alarm")

        Aqf.step("Open CAM Simulation screen in BMS UI.")
        Aqf.step("Set temperature value to 30 and submit.")

        Aqf.checkbox("Verify on the Health display that 'temperature ok' indicator to normal")
        Aqf.checkbox("Verify on the Alarms display that the System_Temperature_Failure "
                     "alarm returns to nominal")

        Aqf.step("Clear the System_Temperature_Failure nominal alarm on the Alarms display")
        Aqf.checkbox("Verify that the System_Temperature_Failure alarm is nominal and "
                     "cleared on the Alarms display")
        Aqf.end()

    @aqf_vr('VR.CM.SITE.OP.17')
    def test_demonstrate_lead_operator_transfer(self):
        """Demonstrate that the lead operator transfer control of the telescope."""
        Aqf.step("Open two KatGUI")
        Aqf.step("Login as 'lead operator' on first KatGUI")
        Aqf.checkbox("Verify that identification and authorisation was required to login")
        Aqf.checkbox("Verify that the current Lead Operator is shown on KatGUI")

        Aqf.step("Login as 'lead operator' on the second KatGUI")
        Aqf.checkbox("Verify that the second KatGUI operator is the new lead operator")
        Aqf.checkbox("Verify that the Lead Operator on the first KatGUI is now logged out")
        Aqf.checkbox("Verify that the Lead Operator can delegate control of a specific subarray to a Control Authority")
        Aqf.checkbox("Verify that the Lead Operator can revoke control of a specific subarray from the Control Authority through assigning control back to LO")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.DO.39')
    def test_demonstrate_cam_allow_one_lead_operator(self):
        """Demonstrate that cam only allow one lead operator."""
        Aqf.step("Open two KatGUIs")

        Aqf.step("Login as 'lead operator' on first KatGUI")
        Aqf.checkbox("Verify that identification and authorisation was required to login")
        Aqf.checkbox("Verify that the current Lead Operator is shown on KatGUI")

        Aqf.step("Login as 'lead operator' on the second KatGUI")
        Aqf.checkbox("Verify that the second KatGUI operator is the new lead operator")
        Aqf.checkbox("Verify that the Lead Operator on the first KatGUI is now logged out")
        Aqf.checkbox("Verify that the Lead Operator can delegate control of a specific subarray to a Control Authority")
        Aqf.checkbox("Verify that the Lead Operator can revoke control of a specific subarray from the Control Authority through assigning control back to LO")

        Aqf.end()

    @aqf_vr('VR.CM.DEMO.L.67')
    def test_demonstrate_clipboard_log_info(self):
        """Demonstrate that cam enables users to copy from any log to the clipboard."""
        Aqf.step("Open USER LOGS display in KatGUI")
        Aqf.step("Select/highlight log text that you want to copy")
        Aqf.checkbox("Copy the log text you highlighted from the user log to the clipboard")
        Aqf.step("Select/highlight any other text you want to copy")
        Aqf.step("Copy the text to the clipboard")
        Aqf.checkbox("Paste the text from the clipboard to the user log")

        Aqf.end()

@system('all')
class TestOperationsPointing(AqfTestCase):
    """Tests for operations pointing display."""

    def setUp(self):
        config = KatObsConfig(self.cam.system)
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        # Set the controlled resource
        self.controlled = specifics.get_controlled_data_proxy(self)
        # Select an antenna and device to use
        self.ant_proxy = self.cam.ants[0].name
        self.db_manager = test_manager.KatObsDbTestManager(config.db_uri)

    def tearDown(self):
        self.cam.subarray_1.req.free_subarray()

    @aqf_vr('VR.CM.DEMO.DP.25')
    def test_system_status_displays_pointing_demo(self):
        """Demonstrate CAM provides system status displays for Pointing Data
        during observations.
        """
        sub_nr = 1
        subarray = getattr(self.cam, 'subarray_{}'.format(sub_nr))
        utils.clear_all_subarrays_and_schedules(self, "Clear test environment")
        sub_obj = getattr(self.cam, "subarray_{}".format(sub_nr))
        Aqf.step("The MeerKAT GUIs have been reworked for timescale B to cover the following Pointing Data requirements")
        Aqf.step("These are:")
        Aqf.step("   1.1) Mechanical pointing for a chosen antenna")
        Aqf.step("   1.2) Commanded pointing position")
        Aqf.step("   1.3) Known pointing error (difference between desired and reported)")
        Aqf.step("   1.4) Electronic pointing of the array  [Note: postponed till multiple steerable beams provided in Correlator]")
        Aqf.step("   1.5) Scan reference position")

        try:
            sb_id1 = utils.create_obs_sb(self, self.ant_proxy, self.controlled,
                                         "CAM_basic_script", runtime=25,
                                         owner='VR.CM.DEMO.DP.25')
            ants_csv, controlled_csv = utils.setup_default_subarray(self, sub_nr,
                antenna_spec=self.ant_proxy, controlled=self.controlled, activate=False)
            Aqf.is_true(subarray.req.set_scheduler_mode("manual").succeeded,
                        "Set scheduler to Manual Scheduling Mode.")
            Aqf.step("Assigning schedule block (%s) "
                 "to subarray %s." % (sb_id1, sub_nr))
            Aqf.step("Assigning schedule block (%s) "
                     "to subarray %s." % (sb_id1, sub_nr))
            schedule_and_check_pass(self, self.db_manager, sub_nr, sb_id1)
            Aqf.step("Activating subarray {} ".format(sub_nr))
            Aqf.is_true(sub_obj.req.activate_subarray(timeout=30).succeeded,
                    "Activation request for subarray {} successful".
                    format(sub_nr))
            Aqf.step("Open the KatGUI subarray1.observations Display")
            Aqf.step("Execute sb {}.".format(sb_id1))
            Aqf.checkbox("Verify that sb {} is executing".format(sb_id1))

            Aqf.step("Hover the mouse on the Pointing Display over %s to monitor the Pointing Data" % self.ant_proxy)

            Aqf.step("1.1) Verify that Mechanical pointing is displayed (actual azim and elev)")
            Aqf.step("1.2) Verify that Commanded pointing position is displayed (requested azim and elev)")
            Aqf.step("1.3) Verify that Known pointing error is displayed (delta-xxx)")
            Aqf.step("1.4) Postponed till multiple steerable beams provided by Correlator")
            Aqf.checkbox("1.5) Verify that Scan Reference Position is displayed (target)")

            Aqf.step("Open KatGUI Sensor Display, click on antenna/receptor {}"
                     " and filter on 'pos'".format(self.ant_proxy))
            Aqf.checkbox("Verify detailed position data is displayed")
            Aqf.step("Stop any executing SBs on subarray {}".format(sub_nr))
            Aqf.checkbox("Verify that sb {} has been stopped".format(sb_id1))

        finally:
            utils.clear_all_subarrays_and_schedules(self, "Cleanup test environment")
        Aqf.end()
