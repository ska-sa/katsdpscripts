###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time

from tests import settings, Aqf, AqfTestCase
from nosekatreport import system, aqf_vr, slow


@system('mkat')
class TestCamVdsFloodlightControl(AqfTestCase):
    """Test that the vds switches on and off through the video
     display subsystem translator"""

    @aqf_vr("VR.CM.AUTO.CV.43")
    @system("mkat")
    def test_vds_floodlights_control(self):
        """Test that the vds floodlights switches on and off through the video
         display subsystem translator """

        Aqf.step("Switch on the floodlights")
        light_on = self.cam.anc.req.vds_floodlight_on("on")
        Aqf.step("Verify that the floodlights are on")
        if light_on.succeeded:
            Aqf.passed("floodlights on")
        else:
            Aqf.failed("floodlights not on")

        Aqf.step("Switch off the floodlights")
        light_off = self.cam.anc.req.vds_floodlight_on("off")
        Aqf.step("Verify that the floodlights are off")
        if light_off.succeeded:
            Aqf.passed("floodlights off")
        else:
            Aqf.failed("floodlights on")

        Aqf.end()


    @aqf_vr("VR.CM.AUTO.CV.44")
    @system("mkat")
    def test_cam_vds_camera_control(self):
        """Test the control of functions of the camera through the video display subsystem"""
        Aqf.step("Pan camera to 70 degress")
        self.cam.anc.req.vds_pan("to", 70)

        Aqf.step("Verify that the camera is panned to 70 degrees")
        pan_sensor = "cam.anc.sensor.vds_pan_position"
        Aqf.sensor(pan_sensor).wait_until(70.0, sleep = 1)
        Aqf.sensor(pan_sensor).eq(70.0)

        Aqf.step("Zoom camera to 50 degrees")
        self.cam.anc.req.vds_zoom( "to", 50)

        Aqf.step("Verify that the camera is  zoomed to 50 degrees")
        zoom_sensor = "cam.anc.sensor.vds_zoom_position"
        Aqf.sensor(zoom_sensor).wait_until(50.0, sleep = 1)
        Aqf.sensor(zoom_sensor).eq(50.0)


        Aqf.step("Select the receptor number(m002) to view")
        self.cam.anc.req.vds_preset_set(2)
        pan_value = self.cam.anc.sensor.vds_pan_position.get_value()
        zoom_value = self.cam.anc.sensor.vds_zoom_position.get_value()

        Aqf.step("Pan the camera to a different direction (60 degrees)")
        self.cam.anc.req.vds_pan("to", 60)

        Aqf.step("Move the camera to receptor 2")
        self.cam.anc.req.vds_preset_goto(2, "set")

        Aqf.step("Verify that the pan position has changed back to %s degrees" %
                 pan_value)
        Aqf.sensor(pan_sensor).wait_until(pan_value, sleep = 1)
        Aqf.sensor(pan_sensor).eq(70.0)

        Aqf.step("Verify that the zoom position has changed back to %s degrees" %
		 zoom_value)
        Aqf.sensor(zoom_sensor).wait_until(zoom_value, sleep = 1)
        Aqf.sensor(zoom_sensor).eq(50.0)

        Aqf.end()
        