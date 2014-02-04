import unittest
import threading
import logging
import time

from katsdpscripts.fake.updater import (WarpClock, PeriodicUpdaterThread,
                                        SingleThreadError)


logging.basicConfig(level=logging.DEBUG)


class TestingUpdate(unittest.TestCase):
    """Run 'nosetests -s --nologcapture' to see output."""
    def setUp(self):
        self.counter = 0

    def update(self, timestamp):
        self.counter += 1
        time.sleep(0.01)
        print 'Updated at', timestamp, 'counter =', self.counter

    def test_slave_sleep(self):
        """Expect to do 1.0 second warp and 1.0 second normal sleep."""
        self.clock = WarpClock(warp=True)
        with PeriodicUpdaterThread([self], self.clock, period=0.1):
            self.clock.slave_sleep(1.0)
            time.sleep(0.5)
            self.clock.warp = False
            self.clock.slave_sleep(0.5)

    def test_overloaded_thread(self):
        """Exercise thread overload warning."""
        self.clock = WarpClock(warp=True)
        with PeriodicUpdaterThread([self], self.clock, period=0.01):
            self.clock.slave_sleep(0.05)

    def bad_sleep(self):
        self.assertRaises(SingleThreadError, self.clock.slave_sleep, 1)

    def test_single_master_slave(self):
        """Check that only a single master and slave thread is allowed."""
        self.clock = WarpClock(warp=True)
        with PeriodicUpdaterThread([self], self.clock, period=0.1):
            # Ensure that master sleep has happened at least once
            self.clock.slave_sleep(0.5)
            self.assertRaises(SingleThreadError, self.clock.master_sleep, 0.5)
            # The pest thread will trigger an exception
            pest_thread = threading.Thread(target=self.bad_sleep, name='PestThread')
            pest_thread.start()
            with self.clock.slave_lock:
                self.assertRaises(SingleThreadError, self.clock.slave_sleep, 1)

    def test_sleep_condition(self):
        """Check that a condition can wake up a sleeping thread."""
        self.clock = WarpClock(warp=True)
        with PeriodicUpdaterThread([self], self.clock, period=0.1):
            satisfied = self.clock.slave_sleep(0.95, condition=lambda: self.counter == 5)
            self.assertTrue(satisfied, 'Sleep condition not satisfied')
            self.assertEquals(self.counter, 5, 'Sleep condition not satisfied')
            satisfied = self.clock.slave_sleep(0.95, condition=lambda: self.counter == 1000)
            self.assertFalse(satisfied, 'Sleep timeout not satisfied')
            self.assertEquals(self.counter, 15, 'Sleep timeout not satisfied')
