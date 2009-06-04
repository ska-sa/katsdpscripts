#!/usr/bin/python
# Print out the list of known catalog objects ordered by elevation

import ffuilib as ffui
import math

observer = ffui.build_observer()
sources = ffui.build_catalog(observer)
 # we dont need access to any devices, so we just
 # build the source catalog manually instead of using
 # cbuild.

sources.update()
 # update sources values to current time

alt_list = []

for source in sources:
    az = sources[source].az.__repr__()
    alt = sources[source].alt.__repr__()
    alt_list.append(alt+":"+az+":"+source)

alt_list.sort()

print "Source Name".center(35),"El (deg)".center(10),"Az (deg)".center(10)
print "".center(35,"="),"".center(10,"="),"".center(10,"=")

for source in alt_list:
    (alt,az,source) = source.split(":")
    alt = "%.2F" % math.degrees(float(alt))
    az = "%.2F" % math.degrees(float(az))
    print str(source).ljust(35),str(alt).center(10),str(az).center(10)

print "Done."
