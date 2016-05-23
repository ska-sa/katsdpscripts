###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import random
import time

from nosekatreport import system, aqf_vr, site_acceptance, site_only
from tests import specifics, fixtures, wait_sensor_includes, wait_sensor_excludes, Aqf, AqfTestCase
from katcorelib.katobslib.common import ScheduleBlockTypes

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
    ok = Aqf.equals(msg.arguments[0], 'ok', ("Request '%s' was sent successfully")
        % msg.name)
    return ok

@system('all')
class TestRemoteOperations(AqfTestCase):
    """Test that CAM detects when backup link is in use and act accordingly"""


    def setUp(self):
        fixtures.sim = self.sim
        fixtures.cam = self.cam
        
        self.sub_nr = 1
        self.sub = self.cam.subarray_1
        self.sub.req.free_subarray()
        self.controlled = specifics.get_controlled_data_proxy(self)
        self.selected_ant = self.cam.ants[-1].name

    def tearDown(self):
        pass


    @aqf_vr("VR.CM.AUTO.I.58")
    def test_cam_action_on_loosing_lo(self):
        """
        Test CAM implements action if triggered on loosing LO connection.
        This action will only be configured when stable operational state is achieved.
        """
        try:
            Aqf.step("Inject an alarm event for loss of Lead Operator")
            self.cam.sys.req.alarm_event("No_Lead_Operator", "set", "automated-stop-observations")
            Aqf.step("Verify the following actions have been implemented:")
            #Aqf.step("Verify system interlock state is XXX")
            #ok = Aqf.sensor(self.cam.sys.sensor.interlock_state).wait_until("automated-stop-observations-xXX", sleep=1, counter=10)
            Aqf.step("Verify antennas are stowed")
            for ant in self.cam.ants:
                ant_proxy = getattr(self.cam, ant.name)
                ok = Aqf.sensor(ant_proxy.sensor.mode).wait_until("STOW", sleep=1, counter=60)
            Aqf.step("Verify schedulers are locked")
            ok = Aqf.sensor(self.cam.sched.sensor.mode_1).wait_until("locked", sleep=1, counter=10)
            Aqf.step("Verify observations are stopped")
            ok = Aqf.sensor(self.cam.sched.sensor.active_schedule_1).wait_until("", sleep=1, counter=30)
            time.sleep(3)
            Aqf.step("Clear the alarm alarm event for loss of Lead Operator")
            self.cam.sys.req.alarm_event("No_Lead_Operator", "cleared", "automated-stop-observations")
            #Aqf.step("Verify schedulers are not locked")
            #ok = Aqf.sensor(self.cam.sched.sensor.mode_1).wait_until("locked", sleep=1, counter=10)
            Aqf.step("Verify system interlock state is back to NONE")
            ok = Aqf.sensor(self.cam.sys.sensor.interlock_state).wait_until("NONE", sleep=1, counter=10)
        finally:
            self.cam.sys.req.operator_resume_operations()
        Aqf.end()

    @aqf_vr("VR.CM.SITE.OPB.9")
    def test_cam_backup_link(self):
        """Test that the CAM stops motion tasks and cron jobs when on backup link"""
        Aqf.step("Open 'Alarm' display in KatGUI.")
        Aqf.step("Unplug/Disconnect main link from kapb to capetown.")
        Aqf.checkbox("Verify that 'System link failure' alarm is triggered.")
        Aqf.step("Open 'Processes Control' display in KatGUI.")
        Aqf.checkbox("Verify that motion processes have stopped.")
        
        Aqf.step("Reconnect main link from KAPB to capetown.")
        Aqf.step("Open 'Processes Control' display in KatGUI.")
        Aqf.checkbox("Verify that motion processes are running.")
        Aqf.step("Verify that 'System link failure' alarm returns to nominal.")
        Aqf.step("Open 'Alarm' display in KatGUI ")
        Aqf.step("Clear 'System link failure' alarm")
        Aqf.checkbox("Verify that 'System link failure' alarm is cleared.")
        
        Aqf.end()

    @aqf_vr('VR.CM.SITE.OP.2')
    def test_demonstrate_control_centre_klerefontein(self):
        """Demonstrate that CAM supports a control centre in Klerefontein."""
        Aqf.step("Open KatGUI in Klerefontein")
        Aqf.step("Login as 'lead operator'")
        Aqf.checkbox("Verify that you are logged in.")
        
        Aqf.end()
    
    @aqf_vr('VR.CM.SITE.OP.3')
    def test_demonstrate_control_centre_losberg(self):
        """Demonstrate that CAM supports a control centre in Losberg."""
        Aqf.step("Open KatGUI in Losberg")
        Aqf.step("Login as 'lead operator'")
        Aqf.checkbox("Verify that you are logged in.")
        
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


    @aqf_vr("VR.CM.SITE.OP.21")
    def test_cam_stop_connectivity_loss(self):
        """Test that the CAM stops observations due to connectivity loss to lead operator at site"""
        Aqf.step("Open KatGUI and login as Operator")
        Aqf.step("Open a second KatGUI on a different operator station and login as Lead Operator")
        Aqf.step("Unplug ethernet cable and turn off wi-fi from 2nd KatGUI station.")
        Aqf.checkbox("Verify that the status bar changes to Red and the there is"
                     " a syncing time error.")
        Aqf.checkbox("On the 1st operator station verify that an alarm is raised indicating that the LO is logged out after the configured time delay.")
                     
        Aqf.step("An alarm is raised when the LO is logged out for more than a configured time.")
        Aqf.step("In the final operational state this alarm will have an action defined to notify katsyscontroller to stop observations.")
        Aqf.step("The final operational state will raise an alarm when LO is logged out for more than a configured time.")
        Aqf.step("This action will not be configured for AR1 integration and commissioning.")
        Aqf.step("Only once stable operational state has been achieved for MeerKAT will this action be uncommented in the alarms configuration.")
        Aqf.end()
