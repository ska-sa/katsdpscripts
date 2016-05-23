###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import random

from nosekatreport import aqf_vr, system
from tests import Aqf, AqfTestCase, specifics


@system('mkat_rts','mkat')
class TestScienceProcessorControl(AqfTestCase):

    """Tests that science processor can be configured and controlled over ethernet
        using katcp protocol."""

    def setUp(self):
        #Get status of installed frequency bands.
        Aqf.step("Setup")
        x = random.randint(0, len(self.cam.ants) - 1)
        self.ant_name = self.cam.ants[x].name
        self.ant_obj = self.cam.ants[x]
        data_name = specifics.get_controlled_data_proxy(self)
        self.data_obj = getattr(self.cam, data_name) #  raises and exception if not found
        self.data_product_id = "array_1_c856M32k"

    def tearDown(self):
        Aqf.step("Return SP to initial state.")
        Aqf.step("Try to deconfigure data product")
        self.data_obj.req.spmc_data_product_configure(self.data_product_id, "")


    def check_if_req_ok(self,obj):
        """Helper function that check if request was successfully sent to device SPMC,
            using katcp protocol."""

        msg = obj.messages[0]
        if msg.arguments[0] == 'ok':
            Aqf.progress("Verify if request '%s' is successfully sent to SPMC."
                % msg.name )
        else:
            #log to the user
            Aqf.progress("Failed to send request '%s' to SPMC." % msg.name)
        ok = Aqf.equals(msg.arguments[0], 'ok', ("Verify if request '%s' is successfully sent"
            " to SPMC using katcp protocol over ethernet.") % msg.name)
        return ok

    @aqf_vr('VR.CM.AUTO.C.23')
    def test_science_processor_sensors_and_requests_exposed(self):
        """Test that science processor sensor and requests are exposed."""

        request_dict = self.data_obj.req.keys()
        Aqf.is_true( len([r for r in request_dict if r.startswith("spmc_")]) > 0,
            "Verify that SPMC has requests.")

        #Check if SP has sensors.
        sp_sensors = self.data_obj.req.sensor_list()
        msg = sp_sensors.messages[1:]
        Aqf.is_true(len([msg[i].arguments[0] for i in range(len(msg))
            if msg[i].arguments[0].startswith("spmc")]) > 0,
                "Verify that SPMC has sensors.")

        #check if SP is synced
        Aqf.equals(self.data_obj.sensor.spmc_state.get_value(), "synced", "Verify that SPMC is synced.")

        Aqf.end()

    @aqf_vr('VR.CM.AUTO.C.23')
    def test_science_processor_configure_control(self):
        """Test that science processor can be configured and controlled."""

        Aqf.step("Verify science processor KATCP requests")
        Aqf.step("Verify that science processor can be controlled using katcp protocol.")
        Aqf.step("Request capture_status")
        self.check_if_req_ok(self.data_obj.req.spmc_capture_status())
        Aqf.step("Request SDP status")
        self.check_if_req_ok(self.data_obj.req.spmc_sdp_status())

        Aqf.end()
