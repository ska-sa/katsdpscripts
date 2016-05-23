###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import random

from nosekatreport import system, aqf_vr
from tests import Aqf, AqfTestCase


@system('mkat','mkat_rts')
class TestReceiverControl(AqfTestCase):
    """Tests that receiver can be controlled over ethernet using katcp protocol."""
    ON_VALUE = True
    OFF_VALUE = False
    def setUp(self):
        Aqf.step("Setup")
        x = random.randint(0, len(self.cam.ants) - 1)
        self.proxy = self.cam.ants[x].name
        self.cam_obj = self.cam.ants[x]

        self.vac_sensor = "cam.%s.sensor.rsc_rsc_vac_pump_running" % self.proxy
        self.comp_sensor = "cam.%s.sensor.rsc_rsc_he_compressor_running" % self.proxy
        self.comp_state_sensor = "cam.%s.sensor.rsc_rsc_he_compressor_state" % self.proxy
        self.vac_state_sensor = "cam.%s.sensor.rsc_rsc_vac_pump_state" % self.proxy
        
        self.vac_value = Aqf.sensor(self.vac_sensor).get()
        self.comp_value = Aqf.sensor(self.comp_sensor).get()

        self.comp_state_value = Aqf.sensor(self.comp_state_sensor).get()
        self.vac_state_value = Aqf.sensor(self.vac_state_sensor).get()

        self.rec_state = Aqf.sensor("cam.%s.sensor.rsc_state" % self.proxy ).get()
        self.bands_status = {"l": None, "s": None, "u": None, "x": None}
        self.get_bands_status

    def tearDown(self):
        Aqf.step("Return rsc to initial state.")
        self.set_bands_status(self.bands_status)

    def check_if_req_ok(self,obj):
        """Helper function that check if request was successfully sent to device,
            using katcp protocol."""

        request_dict=self.cam_obj.req.keys()
        Aqf.is_true(len([r for r in request_dict if r.startswith("rsc_")]) > 0,
            "Verify that receiver has requests.")

        # Check if ap has sensors.
        ap_sensors=self.cam_obj.req.sensor_list()
        msg=ap_sensors.messages[1:]
        Aqf.is_true(len([msg[i].arguments[0] for i in range(len(msg))
            if msg[i].arguments[0].startswith("rsc")]) > 0,
                "Verify if receiver has sensors.")

        # check if ap synced
        Aqf.equals(self.rec_state, "synced", "Verify if receiver is synced.")

        msg = obj.messages[0]
        if msg.arguments[0] == 'ok':
            Aqf.progress("Verify if request '%s' is successfully sent to receiver."
                % msg.name )
        else:
            # log to the user
            Aqf.progress("Failed to send request '%s' to receiver." % msg.name)
        Aqf.equals(msg.arguments[0], 'ok', ("Verify if request '%s' is successfully sent"
            " to receiver using katcp protocol over ethernet.") % msg.name)

    @property
    def get_bands_status(self):
        """Get status of bands"""
        for band in self.bands_status:
            _band_sensor = "cam.%s.sensor.rsc_rx%s_expected_online" % (self.proxy, band)
            if Aqf.sensor(_band_sensor).get() == self.ON_VALUE:
                _value = "true"
            elif Aqf.sensor(_band_sensor).get() == self.OFF_VALUE:
                _value = "false"
            self.bands_status[band] = _value
        return self.bands_status

    def set_bands_status(self, receiver_bands):
        "Helper function to set the status of receivers"
        Aqf.progress("Requesting rsc to value %s" % receiver_bands)

        for band in receiver_bands:
            _sensor = "cam.%s.sensor.rsc_rx%s_expected_online" % (self.proxy, band)
            if band != "ku":
                Aqf.progress("Requesting rsc %s to %s" % (band, receiver_bands[band]))
                r = "rsc_rx%s_expected_online" % band
                req = getattr(self.cam_obj.req, r, None)
                req(receiver_bands[band])
                if receiver_bands[band] == "true":
                    Aqf.progress("Verify if rsc %s is online" % band)
                    Aqf.sensor(_sensor).wait_until(True, sleep=1)
                elif receiver_bands[band] == "false":
                    Aqf.progress("Verify if rsc %s is offline" % band)
                    Aqf.sensor(_sensor).wait_until(False, sleep=1)

    @aqf_vr('VR.CM.AUTO.C.24')
    def test_receiver_control(self):
        """Test if Receiver can be controlled."""
        RECEIVER_BANDS = {"l": "true", "s": "false", "u": "false", "x": "false"}
        
        Aqf.step("Verify if receivers can be controlled over ethernet using katcp protocol.")
        self.check_if_req_ok(self.cam_obj.req.rsc_list_duplex())

        Aqf.step("Verify if receiver bands can be set to online or offline")
        self.set_bands_status(RECEIVER_BANDS)

        Aqf.end()
