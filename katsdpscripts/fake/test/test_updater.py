import unittest
import threading
import logging
import time

from katscripts.updater import SleepWarpClock, PeriodicUpdaterThread


logging.basicConfig(level=logging.DEBUG)


class TestingUpdate(unittest.TestCase):
    """Run 'nosetests -s --nologcapture' to see output."""
    def setUp(self):
        self.clock = SleepWarpClock(warp=True)
        self.updater = PeriodicUpdaterThread([self], self.clock, period=0.1)

    def update(self, timestamp, last_update):
        print 'Updated at', timestamp

    def test_slave_sleep(self):
        """Expect to do 1.0 second warp and 1.0 second normal sleep."""
        self.updater.start()
        self.clock.slave_sleep(1.0)
        time.sleep(0.5)
        self.clock.warp = False
        self.clock.slave_sleep(0.5)
        self.updater.stop()
        self.updater.join()
