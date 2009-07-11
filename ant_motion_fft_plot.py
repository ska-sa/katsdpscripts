#!/usr/bin/env python

# plot fft of desired - actual antenna motion for both axes

import pylab as pl
import ffuilib as ffui
import time
import numpy as np

ff = ffui.tbuild("cfg-telescope.ini","local-ant-only-mot")

# specify target and scan parameters
ff.ant1.req_target_azel(10,30)
ff.ant1.req_scan_sym(2, 2, 20)
ff.ant1.req_mode("POINT")
ff.ant1.wait("lock",True,300)

start_time = time.time()
ff.ant1.req_mode("SCAN")
ff.ant1.wait("scan_status","after",300)
end_time = time.time()

#ra = ff.ant1.sensor_pos_request_scan_ra.value
#dec = ff.ant1.sensor_pos_request_scan_dec.value
#ff.ant1.req_target_radec(ra,dec)
#time.sleep(40.0) # track ra-dec target for a while

# get the sensor history. Assumes that the sensor sampling setup for these in ffuilib config file.
# Note the times for the sensors are not exactly synchronised, so might want
# to improve things by interpolating in future, if necessary
acs_des_azim = ff.ant1.sensor_antenna_acs_desired_azim.get_cached_history(start_time=start_time,end_time=end_time)
acs_des_elev = ff.ant1.sensor_antenna_acs_desired_elev.get_cached_history(start_time=start_time,end_time=end_time)
acs_act_azim = ff.ant1.sensor_antenna_acs_actual_azim.get_cached_history(start_time=start_time,end_time=end_time)
acs_act_elev = ff.ant1.sensor_antenna_acs_actual_elev.get_cached_history(start_time=start_time,end_time=end_time)
req_azim = ff.ant1.sensor_pos_request_scan_azim.get_cached_history(start_time=start_time,end_time=end_time)
req_elev = ff.ant1.sensor_pos_request_scan_elev.get_cached_history(start_time=start_time,end_time=end_time)
act_azim = ff.ant1.sensor_pos_actual_scan_azim.get_cached_history(start_time=start_time,end_time=end_time)
act_elev = ff.ant1.sensor_pos_actual_scan_elev.get_cached_history(start_time=start_time,end_time=end_time)

# disconnect - safer now rather than later in case plot thread issues
#ff.disconnect()

#find smallest list (may be slightly different lengths due to sample timing)
n = min(len(acs_des_azim[0]),len(acs_des_elev[0]),len(acs_act_azim[0]),len(acs_act_elev[0]),
        len(req_azim[0]),len(req_elev[0]),len(act_azim[0]),len(act_elev[0]))

# calc differences
req_act_azim = np.array(req_azim[1][0:n]) - np.array(act_azim[1][0:n])
req_act_elev = np.array(req_elev[1][0:n]) - np.array(act_elev[1][0:n])
des_act_azim = np.array(acs_des_azim[1][0:n]) - np.array(acs_act_azim[1][0:n]) # antenna az offset 45deg
des_act_elev = np.array(acs_des_elev[1][0:n]) - np.array(acs_act_elev[1][0:n])

# plots
pl.figure(0)

#azim
pl.subplot(241)
pl.plot(req_azim[0][0:n],req_azim[1][0:n],'r')
pl.plot(act_azim[0][0:n],act_azim[1][0:n],'y')
pl.ylabel('prox az (req red, act yellow)')

pl.subplot(242)
pl.plot(req_azim[0][0:n],req_act_azim,'r')
pl.plot(acs_des_azim[0][0:n],des_act_azim,'g')
pl.ylabel('(req-act) az (proxy red, ant green)')

pl.subplot(243)
pl.psd(req_act_azim,Fs=10.0)
pl.ylabel('psd: prox (req-act) az')

pl.subplot(244)
pl.psd(des_act_azim,Fs=10.0)
pl.ylabel('psd: ant (req-act) az')

#elev
pl.subplot(245)
pl.plot(req_elev[0][0:n],req_elev[1][0:n],'r')
pl.plot(act_elev[0][0:n],act_elev[1][0:n],'y')
pl.ylabel('prox el (req red, act yellow)')

pl.subplot(246)
pl.plot(req_elev[0][0:n],req_act_elev,'r')
pl.plot(acs_des_elev[0][0:n],des_act_elev,'g')
pl.ylabel('(req-act) el (proxy red, ant green)')

pl.subplot(247)
pl.psd(req_act_elev,Fs=10.0)
pl.ylabel('psd: prox (req-act) el')

pl.subplot(248)
pl.psd(des_act_elev,Fs=10.0)
pl.ylabel('psd: ant (req-act) el')

pl.show()

