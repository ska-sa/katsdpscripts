#!/usr/bin/python
# Drive antenna to two targets and then plot az and el (predefined plot) 

import ffuilib as ffui

ff = ffui.tbuild("cfg-telescope.ini","local_ant_only")
 # make fringe fingder connections

ff.ant2.req_target_azel(20.31,30.45)
 # send an az/el target to antenna 2

ff.ant2.req_mode("POINT")
 # switch to mode point

ff.ant2.wait("lock","1",120)
 # wait for lock to be achieved (timeout=120 seconds)

ff.ant2.req_target_azel(40.2,60.32)
 # send a new az/el target

ff.ant2.wait("lock","1",120)
 # wait for lock again

ff.ant2.plot_az_el()
 # produce a canned plot

raw_input("Hit enter to terminate...")
ff.disconnect()
