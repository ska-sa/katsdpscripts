###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
from nosekatreport import system,  aqf_vr
from tests import Aqf, AqfTestCase

@system('mkat', 'kat7')
class TestPduControl(AqfTestCase):
    """Test that the pdu device switches on and off through the power
    distribution unit translator"""

    def start_up_state(self):
        """Pdus have to be on when the test begins"""
        self.cam.anc.req.pdu1_outlet_on(1, "on")
        self.cam.anc.req.pdu2_outlet_on(2, "on")

    @aqf_vr("CAM_Pdu_Control")
    def test_pdu_power_control(self):
        """Test that the cam software can switch on and off exact outlets of power
        distribution unit"""

        Aqf.step("Switch on pdu outlets - using pdu1 outlet 1 and pdu2 outlet 2")
        self.start_up_state()

        Aqf.step("Verify that the pdu1 outlet8 is on")
        pdu1_outlet1 = self.cam.anc.sensor.pdu1_pdu_outlet1.get_value()
        Aqf.is_true(pdu1_outlet1, "pdu1_outlet1 is on")

        Aqf.step("Verify that the pdu2 outlet2 is on")
        pdu2_outlet2 = self.cam.anc.sensor.pdu2_pdu_outlet2.get_value()
        Aqf.is_true(pdu2_outlet2, "pdu2_outlet2 is on")

        Aqf.step("Powerdown pdu1 outlet 1")
        self.cam.anc.req.pdu1_outlet_on(1, "off")
        Aqf.step("Powerdown pdu2 outlet 2")
        self.cam.anc.req.pdu2_outlet_on(2, "off")

        Aqf.step("Verify that pdu1 outlet1 is off")
        outlet1 = self.cam.anc.sensor.pdu1_pdu_outlet1.get_value()
        Aqf.is_false(outlet1, "Verify pdu1_outlet1 is off")

        Aqf.step("Verify that pdu2 outlet2 is off")
        outlet2 = self.cam.anc.sensor.pdu2_pdu_outlet2.get_value()
        Aqf.is_false(outlet2, "Verify pdu2_outlet2 is off")

        Aqf.step("Switch on pdu outlets")
        self.start_up_state()

        Aqf.end()
