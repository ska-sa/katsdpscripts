#!/usr/bin/python
# Print out the list of known catalog objects ordered by elevation

import ffuilib as ffui
import math
import ConfigParser
from katcore import katconf

aicp = katconf.KatConfig("/var/kat/conf/cfg-user.ini","local_ff_client")
observer = ffui.build_observer(aicp)
sources = ffui.build_catalog(observer, aicp)
 # we dont need access to any devices, so we just
 # build the source catalog manually instead of using
 # cbuild.

print sources.visibility_list()
print "Done."
