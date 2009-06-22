#!/usr/bin/python
# Print out the list of known catalog objects ordered by elevation

import ffuilib as ffui
import math

observer = ffui.build_observer()
sources = ffui.build_catalog(observer)
 # we dont need access to any devices, so we just
 # build the source catalog manually instead of using
 # cbuild.

print sources.visibility_list()
print "Done."
