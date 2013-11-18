###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
import unittest2 as unittest

from katcorelib.testutils import NameSpace

import katcorelib.rts_session as session

class Test_CaptureSessionBase(unittest.TestCase):
    def setUp(self):
        self.DUT = session.CaptureSessionBase()

    def test_get_ant_names(self):
        self.DUT.kat = NameSpace()
        self.DUT.kat.controlled_objects = ['ant1', 'rfe7', 'ant2', 'katarchive']
        self.DUT.kat.__dict__['katconfig'] = NameSpace()
        self.DUT.kat.katconfig.__dict__['arrays'] = {}
        self.DUT.kat.katconfig.arrays = {'ants': ['ant1','ant2']}
        self.assertEqual(self.DUT.get_ant_names(), 'ant1,ant2')

    def test_mkat_get_ant_names(self):
        self.DUT.kat = NameSpace()
        self.DUT.kat.controlled_objects = ['m000', 'rfe7', 'm063', 'katarchive']
        self.DUT.kat.__dict__['katconfig'] = NameSpace()
        self.DUT.kat.katconfig.__dict__['arrays'] = {}
        self.DUT.kat.katconfig.arrays = {'ants': ['m000','m063']}
        self.assertEqual(self.DUT.get_ant_names(), 'm000,m063')
