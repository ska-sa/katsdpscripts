import unittest
import threading
import logging
import time

from katscripts.updater import PeriodicUpdaterThread


logging.basicConfig(level=logging.DEBUG)


class TestingUpdate(unittest.TestCase):
    """Run 'nosetests -s --nologcapture' to see output."""
    def setUp(self):
        self.updater = PeriodicUpdaterThread([self], dry_run=True,
                                             start_time=None, period=0.1)

    def update(self, timestamp):
        print 'Updated at', timestamp

    def test_warp_and_normal_sleep(self):
        """Expect to do 0.5 second warp and 1.0 second normal sleep."""
        self.updater.start()
        self.updater.sleep(0.5)
        time.sleep(0.5)
        self.updater.dry_run = False
        self.updater.sleep(0.5)
        self.updater.stop()
        self.updater.join()
