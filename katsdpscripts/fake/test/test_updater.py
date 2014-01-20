import unittest
import threading
import logging
import time

from katscripts.updater import (SleepWarpClock, PeriodicUpdaterThread,
                                SingleThreadError)


logging.basicConfig(level=logging.DEBUG)


class TestingUpdate(unittest.TestCase):
    """Run 'nosetests -s --nologcapture' to see output."""
    def update(self, timestamp, last_update):
        print 'Updated at', timestamp

    def test_slave_sleep(self):
        """Expect to do 1.0 second warp and 1.0 second normal sleep."""
        self.clock = SleepWarpClock(warp=True)
        with PeriodicUpdaterThread([self], self.clock, period=0.1):
            self.clock.slave_sleep(1.0)
            time.sleep(0.5)
            self.clock.warp = False
            self.clock.slave_sleep(0.5)

    def bad_sleep(self):
        self.assertRaises(SingleThreadError, self.clock.slave_sleep, 1)

    def test_single_master_slave(self):
        """Check that only a single master and slave thread is allowed."""
        self.clock = SleepWarpClock(warp=True)
        with PeriodicUpdaterThread([self], self.clock, period=0.1):
            # Ensure that master sleep has happened at least once
            self.clock.slave_sleep(0.5)
            self.assertRaises(SingleThreadError, self.clock.master_sleep, 0.5)
            # The pest thread will trigger an exception
            pest_thread = threading.Thread(target=self.bad_sleep, name='PestThread')
            pest_thread.start()
            with self.clock.slave_lock:
                self.assertRaises(SingleThreadError, self.clock.slave_sleep, 1)
