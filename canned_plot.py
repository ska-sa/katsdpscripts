#!/usr/bin/python
# Drive antenna to two targets and then plot az and el (predefined plot)

import katuilib as katui

kat = katui.tbuild("cfg-user.ini","local_ant_only")
 # make fringe fingder connections

kat.ant2.req.target_azel(20.31,30.45)
 # send an az/el target to antenna 2

kat.ant2.req.mode("POINT")
 # switch to mode point

kat.ant2.wait("lock","1",120)
 # wait for lock to be achieved (timeout=120 seconds)

kat.ant2.req.target_azel(40.2,60.32)
 # send a new az/el target

kat.ant2.wait("lock","1",120)
 # wait for lock again

kat.ant2.plot_az_el()
 # produce a canned plot

raw_input("Hit enter to terminate...")
kat.disconnect()
