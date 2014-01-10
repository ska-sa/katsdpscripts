import unittest
import threading
import logging

from katscripts import update


logging.basicConfig(level=logging.DEBUG)


class TestingUpdate(unittest.TestCase):
    def setUp(self):
        self.dorm = update.Dormitory()
        self.sleeper = threading.Thread(target=self.sleep, name='SleeperThread')

    def sleep(self):
        print 'Going to bed'
        self.dorm.check_in(1.0)
        print 'Woke up'

    def test_dormitory(self):
        self.sleeper.start()
        for n in range(14):
            print 'Iteration', n + 1
            self.dorm.run(0.1)
        self.sleeper.join()
