
###############################################################################


# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
""" ."""

from tests import settings, Aqf, AqfTestCase
from nosekatreport import system, aqf_vr


if settings.system == "kat7":
    Aqf.progress("Loading kat7 specifics")
    from specifics_kat7 import *
elif settings.system == "mkat_rts":
    Aqf.progress("Loading mkat_rts specifics")
    from specifics_mkat_rts import *
elif settings.system == "mkat":
    Aqf.progress("Loading mkat specifics")
    from specifics_mkat import *
else:
    print "==== UNSUPPORTED SYSTEM:", settings.system

