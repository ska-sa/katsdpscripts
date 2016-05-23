###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
from __future__ import with_statement

import time
import unittest2 as unittest
import re

from nosekatreport import (aqf_vr, system, slow)

from tests import fixtures, wait_sensor, SensorTransitionWaiter
from tests import nodemanager_for, nodemanager_process_req
from tests import nodemanager_process_sensor, Aqf, AqfTestCase


class TestDBE7(AqfTestCase):
    """Test kat7 dbe (DBE7) proxy, i.e. cam.dbe7"""

    def setUp(self):
        #self.cam = fixtures.cam
        #self.sim = fixtures.sim
        pass

    def _get_sensor_list(self):
        replies = self.cam.dbe7.req.sensor_list()
        return [m.arguments[0] for m in replies.messages
                if m.TYPE_NAMES[m.mtype] == 'INFORM']


@system('kat7')
class TestDBE7_sensor_update(TestDBE7, AqfTestCase):
    def tearDown(self):
        # Make sure we don't leave any hidden sensors
        #self.sim.dbe7_dev.req.unhide_sensors('.*')
        pass

    def test_sensor_update(self):
        """Test that the proxy correctly handles device sensors coming and going
        through its #device-changed inform handler
        """
        initial_sensorlist = set(self._get_sensor_list())
        r = self.sim.dbe7_dev.req.hide_sensors('3x.adc.*')
        # Number of sensors that were hidden
        no_hidden = int(r.messages[0].arguments[1])
        #assert(no_hidden > 0)             # Some sensors should have been hidden!
	Aqf.less(0, no_hidden, 'Verify that the number of hidden samples are greater than 0')
        # Names of the sensors that were removed with the dbe prefix added
        removed = set('dbe.'+m.arguments[0] for m in r.messages[1:])
        # sanity check
        Aqf.equals(len(removed), no_hidden, 'Check if removed sensors are equal to the hidden sensors')
	#assert(len(removed) == no_hidden)
        # Expected remapped sensor names with 3x mapping to ant3h
        removed_remapped = set(re.sub('roach\d+.3x', 'ant3h', s)
                               for s in removed)
        time.sleep(0.01)
        # Check that proxy has completed sync with device
        wait_sensor(self.cam.dbe7.sensor.dbe_state, 'synced', timeout=5)
        post_hide_sensorlist = set(self._get_sensor_list())
        # Since we removed sensors with mappings, double the amount of
        # sensors removed from the device should have been removed
        # from the proxy
        Aqf.equals(len(initial_sensorlist) - len(post_hide_sensorlist),
                         2*no_hidden,
                         'Incorrect number of sensors after sensor change')
        # None of the hidden sensors should be around
        Aqf.equals(removed.intersection(post_hide_sensorlist), set(),
                         'Removed sensors still present in sensor list')
        # Nor should any of the remapped sensors be around
        Aqf.equals(removed_remapped.intersection(post_hide_sensorlist),
                         set(),
                         'Removed remapped sensors still present in sensor list')
        # The current sensor list should be the same as the initial
        # list less the removed sensors
        Aqf.equals(post_hide_sensorlist,
                         initial_sensorlist - removed - removed_remapped,
                         'Incorrect sensor list post update')
        # Let's get those sensors back
        self.sim.dbe7_dev.req.unhide_sensors('.*')
        time.sleep(0.01)
        # Check that proxy has completed sync with device
        Aqf.sensor(self.cam.dbe7.sensor.dbe_state).wait_until('synced')
        final_sensorlist = set(self._get_sensor_list())
        # With the hidden sensors added back we should have identical
        # sensor lists, assuming that no sensors were hidden to start
        # with!
        Aqf.equals(initial_sensorlist, final_sensorlist,
                         'Incorrect sensor list after sensors have been '
                         'restored to device')
        Aqf.end()


@system('kat7')
class TestDBE7_hang_during_sync(TestDBE7, AqfTestCase):
    """Test that that proxy will attempt reconnection if device hangs during syncing"""

    def tearDown(self):
        self.sim.dbe7_dev.req.hang_requests(0)

    @slow
    def test_recon(self):
        dbe7 = self.cam.dbe7
        dbe7_dev = self.sim.dbe7_dev
        waiter = SensorTransitionWaiter(
            dbe7.sensor.dbe_state,
            ('synced', 'disconnected', 'syncing',
             'disconnected', 'syncing', 'synced'))
        dbe7_dev.req.hang_requests(1)  # Make requests(apart from watchdog)hang
        dbe7.req.reconnect('dbe')      # Force reconnection with dbe device
        time.sleep(20 * 5)             # 20 missed watchdogs while syncing
        dbe7_dev.req.hang_requests(0)  # Un-hang requests
        waiter.wait(timeout=5)         # Wait for our transition
        Aqf.end()

