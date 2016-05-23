
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time
import random

from tests import settings, Aqf, AqfTestCase
from nosekatreport import aqf_vr, system, slow


class TestCorrelatorControl(AqfTestCase):
    """Test the CAM configuration and control of the MeerKAT correlator."""

    def setUp(self):
        #Get status of installed frequency bands.
        Aqf.step("Setup")
        x = random.randint(0, len(self.cam.ants) - 1)
        self.ant_name = self.cam.ants[x].name
        self.ant_obj = self.cam.ants[x]
        self.data_obj = self.cam.data_1
        self.data_product_id = "c856M4k"
        self.addCleanup(self.free_subarray)
        self.activate_subarray()

    def free_subarray(self):
        Aqf.step('Free the subarray')
        self.check_if_req_ok(self.cam.subarray_1.req.free_subarray())
        Aqf.step("Return CBF to initial state.")

    def check_if_req_ok(self, obj):
        """Helper function that checks if request was successfully sent to the CBF,
            using katcp protocol."""

        msg = obj.messages[0]
        if msg.arguments[0] == 'ok':
            Aqf.progress("Verify that request '%s' was successfully sent."
                % msg.name)
        else:
            #log to the user
            Aqf.progress("Failed to send request '%s'." % msg.name)
        ok = Aqf.equals(msg.arguments[0], 'ok', ("Request '%s' was successfully sent")
            % msg.name)
        return ok

    def activate_subarray(self):
        subarray_active = Aqf.sensor(self.cam.subarray_1.sensor.state)
        Aqf.step('Make sure the cbf is unconfigured initially')
        self.check_if_req_ok(self.cam.subarray_1.req.free_subarray())
        Aqf.step('Now configure an array on subarray_1')
        self.check_if_req_ok(self.cam.subarray_1.req.set_band('l'))
        self.check_if_req_ok(self.cam.subarray_1.req.set_product(self.data_product_id, 1.0))
        self.check_if_req_ok(self.cam.subarray_1.req.assign_resources(
            self.ant_name+',data_1'))
        Aqf.step('Activate subarray_1')
        self.check_if_req_ok(self.cam.subarray_1.req.activate_subarray(timeout=100))
        subarray_active.wait_until('active', sleep=1, counter=10)

    @aqf_vr("VR.CM.AUTO.CB.27")
    @system("mkat", "mkat_rts")
    def test_configure_of_downconversion_freq(self):
        """
        Test the CAM configuration of the downconversion frequency for each CBF
        sub-division.
        """
        Aqf.step('Wait until cbf_state is synced.')
        Aqf.sensor(self.cam.data_1.sensor.cbf_state).wait_until("synced", sleep=1, counter=20)
        data_proxy_centre_freq = Aqf.sensor(self.cam.data_1.sensor.delay_centre_frequency)
        cbf_centre_freq = Aqf.sensor(getattr(self.cam.data_1.sensor,
            'cbf_{}_centerfrequency'.format(self.data_product_id)))

        Aqf.step('Set the centre frequency on the data proxy')
        self.check_if_req_ok(self.cam.data_1.req.set_centre_frequency(
            self.data_product_id, 12345))
        Aqf.equals(data_proxy_centre_freq.get(), 12345,
            'Centre frequency reported by the data proxy matches selection')
        Aqf.equals(cbf_centre_freq.get(), 12345,
            'Centre frequency reported by the CBF matches selection')
        Aqf.end()

    @aqf_vr("VR.CM.AUTO.CB.28")
    @system("mkat", "mkat_rts")
    def test_calc_of_time_delay_polynomial(self):
        """
        Test the CAM calculation of the phase tracking time delay polynomial for each CBF
        sub-division.
        """
        target = Aqf.sensor(self.cam.data_1.sensor.target)
        delays_enabled = Aqf.sensor(self.cam.data_1.sensor.auto_delay_enabled)
        delay_rate = Aqf.sensor(self.cam.data_1.sensor.delays_send_rate)
        delay_rate.wait_until_approximate(0.0)
        Aqf.step('No delays are sent initially')
        self.check_if_req_ok(
            self.cam.data_1.req.auto_delay('on'))
        self.check_if_req_ok(
            self.cam.data_1.req.target_azel('123', '45'))
        Aqf.step("Verify that the sensors match the target")
        Aqf.equals(True, delays_enabled.get(), 'Delays enabled sensor shows True')
        Aqf.equals(target.get(), 'azel, 123:00:00.0, 45:00:00.0',
           'Target sensor has updated')
        Aqf.step('Start capturing - delay updates should be sent')
        self.check_if_req_ok(
                self.cam.data_1.req.capture_init(self.data_product_id))
        self.check_if_req_ok(
                self.cam.data_1.req.capture_start(self.data_product_id))
        delay_rate.wait_until_approximate(35.0, 5.0)
        Aqf.step('Delay rate sensor shows delay updates at ~ 1.7 sec intervals')
        self.check_if_req_ok(
                self.cam.data_1.req.capture_stop(self.data_product_id))
        self.check_if_req_ok(
                self.cam.data_1.req.capture_done(self.data_product_id))
        Aqf.end()

    @aqf_vr("VR.CM.AUTO.CB.29")
    @system("mkat", "mkat_rts")
    def _test_calc_of_complex_gain_correction(self):
        """
        Test the CAM calculation of the per-antenna complex gain correction for each CBF
        sub-division.
        """

        Aqf.waived("WAIVED for QBL(B)")
        Aqf.end()
