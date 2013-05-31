import unittest2 as unittest

from katcorelib.testutils import NameSpace

from katcorelib import session

class Test_CaptureSessionBase(unittest.TestCase):
    def setUp(self):
        self.DUT = session.CaptureSessionBase()

    def test_get_ant_names(self):
        self.DUT.kat = NameSpace()
        self.DUT.kat.controlled_objects = ['ant1', 'rfe7', 'ant2', 'katarchive']
        self.assertEqual(self.DUT.get_ant_names(), 'ant1,ant2')
