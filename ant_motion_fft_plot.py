#!/usr/bin/env python

# plot fft of desired - actual antenna motion for both axes

import pylab as pl
import ffuilib as ffui
import time
import numpy as np

ff = ffui.tbuild("cfg-telescope.ini","local-ant-only-mot")

# specify and drive to a fixed target
ff.ant1.req_target_azel(10,30)
ff.ant1.req_mode("POINT")
ff.ant1.wait("lock","1",300)

# drive to new target, noting start time
start_time = time.time()
time.sleep(4.0) # track start az-el position for a few seconds

ff.ant1.req_target_azel(-5,40)
ff.ant1.wait("lock","1",300)
time.sleep(4.0)

ra = ff.ant1.sensor_pos_request_scan_ra.value
dec = ff.ant1.sensor_pos_request_scan_dec.value
ff.ant1.req_target_radec(ra,dec)
time.sleep(40.0) # track ra-dec target for a while

end_time = time.time()

# get the sensor history. Assumes that the sensor sampling setup for these in ffuilib config file.
# Note the times for the sensors are not exactly synchronised, so might want
# to improve things by interpolating in future, if necessary
acs_des_azim = ff.ant1.sensor_antenna_acs_desired_azim.get_cached_history(start_time=start_time,end_time=end_time)
acs_des_elev = ff.ant1.sensor_antenna_acs_desired_elev.get_cached_history(start_time=start_time,end_time=end_time)
req_azim = ff.ant1.sensor_pos_request_scan_azim.get_cached_history(start_time=start_time,end_time=end_time)
req_elev = ff.ant1.sensor_pos_request_scan_elev.get_cached_history(start_time=start_time,end_time=end_time)
act_azim = ff.ant1.sensor_pos_actual_scan_azim.get_cached_history(start_time=start_time,end_time=end_time)
act_elev = ff.ant1.sensor_pos_actual_scan_elev.get_cached_history(start_time=start_time,end_time=end_time)

#find smallest list (may be slightly different lengths due to sample timing)
n = min(len(acs_des_azim[0]),len(acs_des_elev[0]),len(req_azim[0]),
                  len(req_elev[0]),len(act_azim[0]),len(act_elev[0]))

# disconnect - safer now rather than later in case plot thread issues
ff.disconnect()

# calc differences
des_act_azim = (np.array(acs_des_azim[1][0:n])+45.0) - np.array(act_azim[1][0:n]) # antenna az offset 45deg
des_act_elev = np.array(acs_des_elev[1][0:n]) - np.array(act_elev[1][0:n])
req_act_azim = np.array(req_azim[1][0:n]) - np.array(act_azim[1][0:n])
req_act_elev = np.array(req_elev[1][0:n]) - np.array(act_elev[1][0:n])

# power spectral density plots
pl.figure(0)
pl.subplot(211)
pl.psd(des_act_azim,Fs=10.0)
pl.title("azim (top), elev (bottom) for control-loop desired minus actual")
pl.subplot(212)
pl.psd(des_act_elev,Fs=10.0)

pl.figure(1)
pl.subplot(211)
pl.psd(req_act_azim,Fs=10.0)
pl.title("azim (top), elev (bottom) for proxy req minus actual")
pl.subplot(212)
pl.psd(req_act_elev,Fs=10.0)

# other plots
pl.figure(2)
pl.subplot(211)
pl.plot(req_azim[1],'r')
pl.plot(act_azim[1],'g')
pl.title("proxy req (red) and actual (green) versus time")
pl.ylabel("azim (deg)")
pl.subplot(212)
pl.plot(req_elev[1],'r')
pl.plot(act_elev[1],'g')
pl.ylabel("elev (deg)")
pl.xlabel("time")

pl.figure(3)
pl.subplot(211)
pl.plot(np.real(des_act_azim))
pl.title("(control-loop desired - actual) versus time")
pl.ylabel("azim diff (deg)")
pl.subplot(212)
pl.plot(np.real(des_act_elev))
pl.ylabel("elev diff (deg)")
pl.xlabel("time")

pl.figure(4)
pl.subplot(211)
pl.plot(np.real(req_act_azim))
pl.title("(proxy requested - actual) versus time")
pl.ylabel("azim diff (deg)")
pl.subplot(212)
pl.plot(np.real(req_act_elev))
pl.ylabel("elev diff (deg)")
pl.xlabel("time")

pl.show()

