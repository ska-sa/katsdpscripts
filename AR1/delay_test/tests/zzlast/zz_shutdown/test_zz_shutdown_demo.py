###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time

import katconf

from nosekatreport import (aqf_vr, system,
                            site_only, site_acceptance)
from tests import Aqf, AqfTestCase, utils, specifics

from katcorelib.katobslib.common import ScheduleBlockTypes


@system("all")
class TestZzShutdown(AqfTestCase):

    """Tests for Operator shutdown computing."""

    def setUp(self):
        self.controlled = specifics.get_controlled_data_proxy(self)
        pass

    def tearDown(self):
        pass

    def _wait_sensor_value_contains(self, katsensor, look_for, timeout):
        """Test katsensor value every second to see if it contains a string, until timeout"""
        i = 0
        while i < timeout:
            val = katsensor.get_value()
            if look_for in str(val):
                return True
            i = i + 1
            time.sleep(1.0)
        return False

    def _create_and_schedule_obs_sb(self, ant_id, owner="aqf-test"):
        Aqf.step("Creating an OBSERVATION SB for receptor/antenna %s" % ant_id)
        sb_id_code = self.obs.sb.new(owner=owner, antenna_spec=ant_id, controlled_resources=self.controlled)
        self.obs.sb.description = "Point source scan for %s" % ant_id
        self.obs.sb.type = ScheduleBlockTypes.OBSERVATION
        # quick scan over the sun
        self.obs.sb.instruction_set = "run-obs-script ~/scripts/observation/point_source_scan.py 'SUN'"
        self.obs.sb.schedule(1)
        Aqf.step("Wait for SB (%s) to complete verifying" % sb_id_code)
        ok = self._wait_sensor_value_contains(self.cam.sched.sensor.observation_schedule_1, sb_id_code, 180)
        Aqf.checkbox("Verify the newly created OBSERVATION SB (%s) for %s "
            "in the Observation Schedule table" % (sb_id_code, ant_id))
        return sb_id_code

    @aqf_vr('VR.CM.DEMO.OO.59')
    def test_zz1_power_down_sp_and_corr(self):
        """Demonstrate that CAM provides an interface to power down SP and Corr for the lead operator on KatGUI.
        """
        # Lead operator to remotely power down SDP and CBF
        Aqf.step("NOTE: This test will shutdown the Science Processor and Correlator!!!")
        Aqf.step("      These will require manual restart of SP and CBF for the AQF to continue correctly.")
        Aqf.step("      If you don't want to do that now, then do not follow the prescribed actions.")

        Aqf.step("Verify that the PDU configurations for SP and CBF are correct in katconfig, including powerdown delays")
        Aqf.keywait("Continue when OK")

        Aqf.step("Open the KATGUI Sensor List display")
        Aqf.step("Filter MCP on 'connected' and verify that the CMC is connected")
        Aqf.step("Filter DATA_1 on 'connected' and verify that the SPMC is connected")
        Aqf.step("Filter ANC on 'outlet' and verify that all PDU outlets are on")
        Aqf.checkbox("The CMC and SPMC are running and their PDU outlets switched on")

        Aqf.step("Open the KATGUI Operator Controls display")
        Aqf.step("Select 'Power down SP and Corr' from the KATGUI")

        Aqf.step("Open the KATGUI Sensor List display")
        Aqf.step("Filter MCP on 'connected' and verify that the CMC is now disconnected")
        Aqf.step("Filter DATA_1 on 'connected' and verify that the SPMC is now disconnected")
        Aqf.step("Filter ANC on 'outlet' and verify that two PDU outlets are now off")
        Aqf.checkbox("The CMC and SPMC have stopped running and there PDU outlets have been switched off")

        Aqf.step("Open 'System Logging' display.")
        Aqf.checkbox("Verify that powerdown activity is logged in 'activity.log'")

        Aqf.step("Restart (kill and start) the stopped simulators manually from iPython")
        Aqf.step("configure_cam(...); cam.status()")
        Aqf.step("e.g. cam.nm_sim/proxy.req.kill('spmc')")
        Aqf.step("     cam.nm_sim/proxy.req.start('spmc')")
        Aqf.step("     cam.nm_sim/proxy.req.kill('cmc')")
        Aqf.step("    cam.nm_sim/proxy.req.start('cmc')")
        Aqf.step("Switch on the two PDU outlets that were switched off")
        Aqf.step("e.g. cam.anc.req.pdu1_outlet_on(8, 'on')")
        Aqf.step("     cam.anc.req.pdu2_outlet_on(16, 'on')")
        Aqf.checkbox("Verify that the Science Processor and Correlator master controllers are running - configure_sim(); sim.status()")
        Aqf.checkbox("Verify that SP and Correlator return to fully operational state")
        Aqf.checkbox("Select 'RESUME OPERATIONS' from Operator Control display")

        Aqf.end()

    @site_only
    @aqf_vr('VR.CM.SITE.OP.20')
    def test_zz1_remote_power_down_sp_and_corr_on_site(self):
        """Demonstrate that CAM provides an interface to power down SP and Corr remotely.
        """
        # Operators to remotely power down SDP and CBF
        Aqf.step("NOTE: This test will shutdown the Science Processor and Correlator!!!")
        Aqf.step("      These will require manual restart of SP and CBF for the AQF to continue correctly.")
        Aqf.step("      If you don't want to do that now, then do not follow the prescribed actions.")

        Aqf.step("Verify that the PDU configurations for SP and CBF are correct in katconfig, including powerdown delays")
        Aqf.keywait("Continue when OK")

        Aqf.step("Open KatGUI and login as Lead Operator")
        Aqf.step("Open 'OPERATOR CONTROL' display.")
        Aqf.step("Click 'POWERDOWN SP and CORR' button.")
        Aqf.checkbox("Verify that SP and Correlator are powered down.")
        Aqf.step("Open 'System Logging' display.")
        Aqf.checkbox("Verify that powerdown activity is logged in 'activity.log'")
        Aqf.step("Manually power up SP")
        Aqf.step("Manually power up Correlator")
        Aqf.checkbox("Verify that SP and Correlator return to fully operational state")
        Aqf.checkbox("Select 'RESUME OPERATIONS' from Operator Control display")
        Aqf.end()

    @aqf_vr('VR.CM.DEMO.OI.4')
    def test_zz2_shutdown_computing(self):
        """Demonstrate that CAM provides an interface to shutdown Computing for the operators on KatGUI.
        """

        # Operators to shutdown computing
        Aqf.step("NOTE: This test will shutdown the Science Processor, Correlator and CAM!!!")
        Aqf.step("      It will require manual restart of the systems for the AQF to continue correctly.")
        Aqf.step("      If you don't want to do that now, then do not follow the prescribed actions.")

        Aqf.step("Verify that the PDU configurations for CAM, SP and CBF are correct in katconfig, including powerdown delays")
        Aqf.keywait("Continue when OK")

        Aqf.step("Open KatGUI")
        Aqf.step("Login as 'operator' in KatGUI")
        Aqf.step("Verify that you are logged in.")
        Aqf.step("Open 'OPERATOR CONTROL' display.")
        Aqf.step("Click 'SHUTDOWN COMPUTING' button.")
        Aqf.step("Note - the dev/lab environments do not actually shutdown the CAM through PDUs but does halt the nodes.")
        Aqf.step("Open 'System Logging' display.")
        Aqf.checkbox("Verify that powerdown activity is logged in 'activity.log'")
        Aqf.checkbox("Verify that SP, Correlator and CAM are powered down.")
        Aqf.step("Manually power up and restart SP, Correlator and CAM - if required")
        Aqf.checkbox("Verify that SP, Correlator and CAM return to fully operational state")
        Aqf.end()


