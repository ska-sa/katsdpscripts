#!/usr/bin/env python
# Drive antenna 1 in various ways and make plots of the motion

# startup ant proxies: kat-launch.py -i cfg-telescope.ini -s ant_only_moog1_office
#                  or: kat-launch.py -i cfg-telescope.ini -s local_ant_only

import ffuilib as ffui
import matplotlib.pyplot as pl
import time
import numpy as np

# from array.ini file
antenna_az_offset_deg = -5.0

# select the type of antenna motion (ensure antenna and control PC NTP synced)
motions = ['az-el scan','az-el pointing','ra-dec track','GPS track','slew']
#motions = ['az-el scan','ra-dec track']
#motions = ['az-el scan']
#motions = ['GPS track']

def make_plots(ff,start_time,end_time,title,fig_num):

    # Get the sensor history. Assumes that the sensor sampling setup for these in build config.
    # Note the times for the sensors are not exactly synchronised, so might want
    # to improve things in the difference calcs by interpolating in future, if necessary

    acs_des_azim = ff.ant1.sensor_antenna_acs_desired_azim.get_cached_history(start_time=start_time,end_time=end_time)
    acs_des_elev = ff.ant1.sensor_antenna_acs_desired_elev.get_cached_history(start_time=start_time,end_time=end_time)
    acs_act_azim = ff.ant1.sensor_antenna_acs_actual_azim.get_cached_history(start_time=start_time,end_time=end_time)
    acs_act_elev = ff.ant1.sensor_antenna_acs_actual_elev.get_cached_history(start_time=start_time,end_time=end_time)
    req_azim = ff.ant1.sensor_pos_request_scan_azim.get_cached_history(start_time=start_time,end_time=end_time)
    req_elev = ff.ant1.sensor_pos_request_scan_elev.get_cached_history(start_time=start_time,end_time=end_time)
    act_azim = ff.ant1.sensor_pos_actual_scan_azim.get_cached_history(start_time=start_time,end_time=end_time)
    act_elev = ff.ant1.sensor_pos_actual_scan_elev.get_cached_history(start_time=start_time,end_time=end_time)

    #find smallest list (may be slightly different lengths due to sample timing)
    n = min(len(acs_des_azim[0]),len(acs_des_elev[0]),len(acs_act_azim[0]),len(acs_act_elev[0]),
            len(req_azim[0]),len(req_elev[0]),len(act_azim[0]),len(act_elev[0]))

    # calc differences
    req_act_azim = np.array(req_azim[1][0:n]) - np.array(act_azim[1][0:n])
    req_act_elev = np.array(req_elev[1][0:n]) - np.array(act_elev[1][0:n])
    des_act_azim = np.array(acs_des_azim[1][0:n]) - np.array(acs_act_azim[1][0:n]) # antenna az offset 45deg
    des_act_elev = np.array(acs_des_elev[1][0:n]) - np.array(acs_act_elev[1][0:n])

    # plots
    fig = pl.figure(fig_num)
    fig.text(0.1,0.95,title + ' (azim - top, elev - bottom)')

    #azim
    pl.subplot(241)
    pl.plot(req_azim[0][0:n],req_azim[1][0:n],'r')
    pl.plot(act_azim[0][0:n],act_azim[1][0:n],'y')
    pl.plot(acs_des_azim[0][0:n],np.array(acs_des_azim[1][0:n])+antenna_az_offset_deg,'g')
    pl.plot(acs_act_azim[0][0:n],np.array(acs_act_azim[1][0:n])+antenna_az_offset_deg,'b')
    pl.title('req,act: prox r,y; ant g,b',fontsize='small')

    pl.subplot(242)
    pl.plot(req_azim[0][0:n],req_act_azim,'r')
    pl.plot(acs_des_azim[0][0:n],des_act_azim,'g')
    pl.title('(req-act) prox r, ant g',fontsize='small')

    pl.subplot(243)
    pl.psd(req_act_azim,Fs=10.0)
    pl.title('psd: prox (req-act)',fontsize='small')

    pl.subplot(244)
    pl.psd(des_act_azim,Fs=10.0)
    pl.title('psd: ant (req-act)',fontsize='small')

    #elev
    pl.subplot(245)
    pl.plot(req_elev[0][0:n],req_elev[1][0:n],'r')
    pl.plot(act_elev[0][0:n],act_elev[1][0:n],'y')
    pl.plot(acs_des_elev[0][0:n],acs_des_elev[1][0:n],'g')
    pl.plot(acs_act_elev[0][0:n],acs_act_elev[1][0:n],'b')
    pl.title('req, act: prox r,y; ant g,b',fontsize='small')

    pl.subplot(246)
    pl.plot(req_elev[0][0:n],req_act_elev,'r')
    pl.plot(acs_des_elev[0][0:n],des_act_elev,'g')
    pl.title('(req-act) prox r, ant g',fontsize='small')

    pl.subplot(247)
    pl.psd(req_act_elev,Fs=10.0)
    pl.title('psd: prox (req-act)',fontsize='small')

    pl.subplot(248)
    pl.psd(des_act_elev,Fs=10.0)
    pl.title('psd: ant (req-act)',fontsize='small')


if __name__ == '__main__':

    ff = ffui.tbuild("cfg-user-pointing.ini","local_ant_only")

    for motion in motions:
        if motion == 'az-el scan':
            print 'performing az-el scan'
            ff.ant1.req_target_azel(10,30)
            ff.ant1.req_scan_sym(4, 4, 20)
            ff.ant1.req_mode("POINT")
            ff.ant1.wait("lock",True,300)
            start_time = time.time()
            ff.ant1.req_mode("SCAN")
            ff.ant1.wait("scan_status","after",300)
            end_time = time.time()
            make_plots(ff,start_time,end_time,'az-el scan',0)
        elif motion == 'az-el pointing':
            print 'tracking az-el target'
            ff.ant1.req_target_azel(-10,30)
            ff.ant1.req_mode("POINT")
            ff.ant1.wait("lock",True,300)
            start_time = time.time()
            time.sleep(40.0)
            end_time = time.time()
            make_plots(ff,start_time,end_time,'az-el pointing',1)
        elif motion == 'ra-dec track':
            print 'tracking ra-dec target'
            ff.ant1.req_target_azel(-10,30)
            ff.ant1.req_mode("POINT")
            ff.ant1.wait("lock",True,300)
            ra = ff.ant1.sensor_pos_request_scan_ra.value
            dec = ff.ant1.sensor_pos_request_scan_dec.value
            ff.ant1.req_target_radec(ra,dec)
            start_time = time.time()
            time.sleep(40.0)
            end_time = time.time()
            make_plots(ff,start_time,end_time,'ra-dec track',2)
        elif motion == 'GPS track':
            print 'tracking GPS satellite'
            cat = ff.sources.filter(tags=['GPS'],el_limit_deg=[20,80])
            tgt = [t for t in cat][0] # get one target
            ff.ant1.req_target(tgt.description)
            ff.ant1.req_mode("POINT")
            ff.ant1.wait("lock",True,300)
            start_time = time.time()
            time.sleep(40.0)
            end_time = time.time()
            make_plots(ff,start_time,end_time,'GPS track',3)
        elif motion == 'slew':
            print 'slewing'
            ff.ant1.req_target_azel(10,30)
            ff.ant1.req_mode("POINT")
            ff.ant1.wait("lock",True,300)
            start_time = time.time()
            ff.ant1.req_target_azel(-100,80)
            ff.ant1.wait("lock",True,300)
            end_time = time.time()
            make_plots(ff,start_time,end_time,'slewing',4)

    pl.show()
    ff.disconnect() # clean up
