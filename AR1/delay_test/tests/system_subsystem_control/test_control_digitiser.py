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
from tests import Aqf, AqfTestCase


@system('mkat_rts','mkat')
class TestDigitiserControl(AqfTestCase):

    """Tests that digitiser can be configured and controlled over ethernet
        using katcp protocol."""

    def setUp(self):
        #Get status of intalled frequency bands.
        Aqf.step("Setup")

        x = random.randint(0, len(self.cam.ants) - 1)
        self.ant_obj = self.cam.ants[x]
        ant_proxy_name = self.ant_obj.name

        self.dig_state = Aqf.sensor("cam.%s.sensor.dig_state" % ant_proxy_name ).get()

        self.dig_obj = getattr(self.cam, ant_proxy_name)
        self.band_args_status = (
            self.dig_obj.req.dig_digitiser_status('l').messages[0].arguments)
        self.rpl, self.bnd, self.status, self.srl_no = self.band_args_status

        #sensor.rscadmin_state.get_value

    def tearDown(self):
        Aqf.step("Return the digitiser frequency band to the state it was at "
            "when we start test.")
        self.dig_obj.req.dig_digitiser_status(self.bnd, self.status, self.srl_no)

    def check_if_req_ok(self,obj):
        """Helper function that check if request was successfully sent to device,
            using katcp protocol."""

        request_dict = self.ant_obj.req.keys()
        Aqf.is_true( len([r for r in request_dict if r.startswith("dig_")]) > 0,
            "Verify that digitiser has requests.")

        #Check if ap has sensors.
        ap_sensors=self.ant_obj.req.sensor_list()
        msg=ap_sensors.messages[1:]
        Aqf.is_true(len([msg[i].arguments[0] for i in range(len(msg))
            if msg[i].arguments[0].startswith("dig")]) > 0,
                "Verify if digitiser has sensors.")

        #check if ap synced
        Aqf.equals(self.dig_state, "synced", "Verify if digitiser is synced.")

        msg = obj.messages[0]
        if msg.arguments[0] == 'ok':
            Aqf.progress("Verify if request '%s' is successfully sent to digitiser."
                % msg.name )
        else:
            #log to the user
            Aqf.progress("Failed to send request '%s' to digitiser." % msg.name)
        Aqf.equals(msg.arguments[0], 'ok', ("Verify if request '%s' is successfully sent"
            " to digitiser using katcp protocol over ethernet.") % msg.name)

    @aqf_vr('VR.CM.AUTO.C.25')
    def test_digitiser_configure_control(self):
        """Test if digitiser can be configured and controlled."""

        Aqf.step("Verify if digitiser can be configured.")
        Aqf.progress("Requesting digitiser to set installed frequency bands")
        _status = ["ready", "standby", "absent"]

        for stat in [x for x in _status if x != self.status]:
            Aqf.progress("Requesting digitiser to set status of %s bands to %s" %
                (self.bnd, stat))
            self.ant_obj.req.dig_digitiser_status(self.bnd, stat, self.srl_no)
            Aqf.equals(stat,
                self.ant_obj.req.dig_digitiser_status(self.bnd).messages[0].arguments[2],
                    "Verify if the status of '%s' band is successfully set to %s status" %
                        (self.bnd, stat))

        Aqf.step("Verify if digitiser can be controlled over ethernet using katcp protocol.")
        self.check_if_req_ok(self.ant_obj.req.dig_capture_start("host"))


        Aqf.end()