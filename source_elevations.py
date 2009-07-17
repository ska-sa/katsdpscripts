#!/usr/bin/python
# Print out the list of known catalog objects ordered by elevation

import ffuilib as ffui
import math
import ConfigParser

aicp = ConfigParser.RawConfigParser()
aicp.read("/var/kat/conf/cfg-user.ini")
observer = ffui.build_observer(aicp, "default_sources")
sources = ffui.build_catalog(observer)
 # we dont need access to any devices, so we just
 # build the source catalog manually instead of using
 # cbuild.

print sources.visibility_list()
print "Done."
