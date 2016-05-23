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


@system('all')
class TestCBFControl(AqfTestCase):

    """Tests that correlator can be configured and controlled."""

    def setUp(self):
        #Get status of installed frequency bands.
        Aqf.step("Setup")
        x = random.randint(0, len(self.cam.ants) - 1)
        self.ant_name = self.cam.ants[x].name
        self.ant_obj = self.cam.ants[x]
        self.data_product_id = "c856M4k"
        self.controlled = "data"
        self.specific_controlled = specifics.get_specific_controlled(self, sub_nr=1, controlled=self.controlled)  # e.g. data_n or dbe7
        self.cam_data_proxy = getattr(self.cam, self.specific_controlled, None)
        self.addCleanup(self.free_subarray)
        self.activate_subarray()

    def free_subarray(self):
        Aqf.step('Free the subarray')
        self.check_if_req_ok(self.cam.subarray_1.req.free_subarray())
        Aqf.step("Return CBF to initial state.")

    def check_if_req_ok(self, obj):
        """Helper function that checks if request was successfully sent,
            using katcp protocol."""

        msg = obj.messages[0]
        if msg.arguments[0] == 'ok':
            Aqf.progress("Verify that request '%s' is successfully sent."
                % msg.name)
        else:
            #log to the user
            Aqf.progress("Failed to send request '%s' to CBF." % msg.name)
        ok = Aqf.equals(msg.arguments[0], 'ok', ("Request '%s' was successfully sent"
            " ") % msg.name)
        return ok

    def activate_subarray(self):
        subarray_active = Aqf.sensor(self.cam.subarray_1.sensor.state)
        Aqf.step('Make sure the cbf is unconfigured initially')
        self.check_if_req_ok(self.cam.subarray_1.req.free_subarray())
        Aqf.step('Now configure an array')
        self.check_if_req_ok(self.cam.subarray_1.req.set_band('l'))
        self.check_if_req_ok(self.cam.subarray_1.req.set_product(self.data_product_id, 1.0))
        self.check_if_req_ok(self.cam.subarray_1.req.assign_resources(
            self.ant_name + ','+self.specific_controlled))
        Aqf.step('Activate the array')
        self.check_if_req_ok(self.cam.subarray_1.req.activate_subarray(timeout=100))
        subarray_active.wait_until('active', sleep=1, counter=10)

    @system('mkat_rts', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.CB.30')
    def test_cmc_sensors_and_requests_exposed(self):
        """Test that CMC sensor and requests are exposed."""

        #check if CMC is synced
        Aqf.equals(self.cam.mcp.sensor.cmc_state.get_value(), "synced", "Verify that CMC is synced.")

        request_dict = self.cam.mcp.req.keys()
        Aqf.is_true(any([r for r in request_dict if r.startswith("cmc_")]),
            "Verify that CMC has requests.")

        #Check if CMC has sensors.
        sensors = self.cam.mcp.req.sensor_list()
        msgs = sensors.messages[1:]
        Aqf.is_true(any([msg.arguments[0] for msg in msgs if msg.arguments[0].startswith("cmc")]),
                    "Verify that CMC has sensors.")
        Aqf.step("Request resource list from CMC.")
        self.check_if_req_ok(
            self.cam.mcp.req.cmc_resource_list())

        Aqf.step("Request array list from CMC.")
        self.check_if_req_ok(
            self.cam.mcp.req.cmc_array_list())

        Aqf.end()

    @system('mkat_rts', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.CB.26')
    def test_cam_calculate_doppler_corrections(self):
        """Test that CAM calculates doppler corrections and enables/disables doppler."""

        Aqf.waived("WAIVED FOR QBL(B) - Verify that CAM data proxies calculate doppler corrections and send to CBF")
        Aqf.waived("WAIVED FOR QBL(B) - Verify that CAM data proxies can disable doppler offset")
        Aqf.end()

    @system('mkat_rts', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.CB.31')
    def test_cam_calculate_and_apply_polarisation_corrections(self):
        """Test that CAM calculates and applies polarisation corrections."""

        Aqf.waived("WAIVED FOR QBL(B) - Verify that CAM data proxies calculate and apply polarisation corrections and send to CBF")
        Aqf.end()

    @system('mkat_rts', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.CB.30')
    def test_cam_apply_steering_offset_multibeam(self):
        """Test that CAM applies a steering offset to the main beam."""

        Aqf.waived("WAIVED FOR QBL(B) - Multibeam support not required for AR1")
        Aqf.end()

    @system('mkat_rts', 'mkat', all=False)
    @aqf_vr('VR.CM.AUTO.CB.30')
    def test_cam_apply_steering_offset(self):
        """Test that CAM applies a steering offset to the main beam."""
        Aqf.skipped(
            "Beam offset has been temporarily disabled in KATpoint - see JIRA CB-717")
        Aqf.end()
        return

        base_az = Aqf.sensor(self.cam_data_proxy.sensor.pos_request_base_azim)
        base_el = Aqf.sensor(self.cam_data_proxy.sensor.pos_request_base_elev)
        offset_az = Aqf.sensor(self.cam_data_proxy.sensor.pos_request_offset_azim)
        offset_el = Aqf.sensor(self.cam_data_proxy.sensor.pos_request_offset_elev)
        offset_x = Aqf.sensor(self.cam_data_proxy.sensor.offset_fixed_x)
        offset_y = Aqf.sensor(self.cam_data_proxy.sensor.offset_fixed_y)
        offset_proj = Aqf.sensor(self.cam_data_proxy.sensor.offset_fixed_projection)
        offset_time = Aqf.sensor(self.cam_data_proxy.sensor.offset_time)
        delays_enabled = Aqf.sensor(self.cam_data_proxy.sensor.auto_delay_enabled)
        target = Aqf.sensor(self.cam_data_proxy.sensor.target)

        Aqf.step("Start with zero offset")
        self.check_if_req_ok(
            self.cam_data_proxy.req.offset_fixed('0', '0'))
        Aqf.step("Set an (az, el) target via the data proxy")
        self.check_if_req_ok(
            self.cam_data_proxy.req.target_azel('123', '45'))
        Aqf.equals(target.get(),'azel, 123:00:00.0, 45:00:00.0',
                'Target sensor has updated')
        Aqf.step("enable auto delay updates to CBF")
        self.check_if_req_ok(
            self.cam_data_proxy.req.auto_delay('on'))
        Aqf.equals(delays_enabled.get(), True, 'Delays enabled sensor shows True')
        Aqf.step("Verify that the sensors match the target")
        base_az.wait_until_approximate(123.0)
        base_el.wait_until_approximate(45.0)
        offset_x.wait_until_approximate(0.0)
        offset_y.wait_until_approximate(0.0)

        Aqf.step("Now start capture, and delay updates should start too")
        self.check_if_req_ok(
            self.cam_data_proxy.req.capture_start('c856M4k'))

        Aqf.wait(5.0, 'Wait for the delay updates to start')
        offset_az.wait_until_approximate(123.0)
        offset_el.wait_until_approximate(45.0)
        Aqf.step("Set a fixed offset")
        self.check_if_req_ok(
            self.cam_data_proxy.req.offset_fixed('3', '4', 'gnomonic'))
        Aqf.step("Verify that the base az. el sensors are unaffected")
        Aqf.equals(base_az.get(), 123.0, ' Base azimuth unchanged')
        Aqf.equals(base_el.get(), 45.0, 'Base elevation unchanged')
        Aqf.step("Verify that the fixed offset sensors match the request")
        Aqf.equals(offset_x.get(), 3.0,  'Fixed offset x is correct')
        Aqf.equals(offset_y.get(), 4.0, 'Fixed offset y is correct')
        Aqf.equals(offset_proj.get(), 'gnomonic', 'Fixed offset projection is correct')
        Aqf.wait(5.0, 'Wait for more delay updates')
        Aqf.almost_equals(offset_az.get(), 123.0 + 3.0, 2.0, 'Azimuth offset has been applied')
        Aqf.almost_equals(offset_el.get(), 45.0 + 4.0, 2.0, 'Elevation offset has been applied')

        Aqf.step("Stop capture")
        self.check_if_req_ok(
            self.cam_data_proxy.req.capture_stop('c856M4k'))
        Aqf.end()
