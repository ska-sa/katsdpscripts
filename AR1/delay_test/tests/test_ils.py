###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import time

from tests import fixtures, wait_with_strategy, Aqf, AqfTestCase
from nosekatreport import (aqf_vr, system)
from .utils import execute


@system('kat7')
class TestIls(AqfTestCase):

    """Tests ils."""

    def setUp(self):
        pass
        #self.sim.asc.req.set_sensor_value('fire.ok', 1)

    def tearDown(self):
        #self.sim.asc.req.set_sensor_value('fire.ok', 1)
        pass

    @aqf_vr('CAM_ILS_reporting')
    @system('kat7')
    def test_asc_fire_not_ok(self):
        """Send update to ILS."""
        #self.cam.anc.sensor.asc_fire_ok.set_strategy('event')
        #self.assertEqual(self.cam.anc.sensor.asc_fire_ok.status, 'nominal')

        #Trigger the fire sensor
        #self.sim.asc.req.set_sensor_value('fire.ok', 0)
        # Wait because kataware uses a cyclic processing
        Aqf.step("Set up")
        time.sleep(3)
        ils_metric = "anc.asc.fire.ok"
        # | grep  "Success sending ILS metric" | grep ""%s""' % (ils_metric))
        ok, output = execute('tail /var/kat/log/ils.log')
        print "+++", ok, output
        self.assertTrue(ok,
                        "Could not execute tail for ils.log for metric %s" %
                        (ils_metric))
        self.assertTrue(output[0] is not None,
                        "Could not find tail for ils.log")
        success_output = [s for s in output[0].splitlines()
                          if "Success sending ILS metric" in s
                          and ils_metric in s]
        self.assertTrue(success_output is not None,
                        "Could not find log for successful sending "
                        "of metric %s" % (ils_metric))
        Aqf.passed("Successfully sent ILS metric %s" % ils_metric)
        #Reset the sensor
        #self.sim.asc.req.set_sensor_value('fire.ok', 1)
        #TBD - Use katlogserver to check alarms.log (grep-log or tail-log)
        Aqf.end()
#
