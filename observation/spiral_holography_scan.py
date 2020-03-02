#!/usr/bin/env python
# Perform spiral holography scan on specified target(s). Mostly used for beam pattern measurement.
#
# to run on simulator:
# ssh kat@monctl.comm
# kat-start.sh (may need to call kat-stop.sh, or better kat-kill.py, and possibly kill all screen sessions on kat@monctl.comm, and possibly kat@proxy.monctl.comm)
# ipython
# import katuilib
# configure()
# %run ~/scripts/observation/spiral_holography_scan.py -f 1722 -a ant1,ant2,ant3,ant4,ant5,ant6,ant7 -b ant2,ant3 --num-cycles 1 --cycle-duration 120 -l 12 'AFRISTAR' -o mattieu --sb-id-code='20121030-0003'
# look on http://kat-flap.control.kat.ac.za/kat/KatGUI.swf and connect to 'comm'
#
#using schedule blocks
#help on schedule blocks: https://sites.google.com/a/ska.ac.za/intranet/teams/operators/kat-7-nominal-procedures/frequent-tasks/control-tasks/observation
#to view progress: http://192.168.193.8:8081/tailtask/<sb_id_code>/progress
#to view signal displays remotely safari goto "vnc://kat@right-paw.control.kat.ac.za"
#
#ssh kat@kat-ops.karoo
#ipython
#import katuilib
#configure_obs()
#obs.sb.new_clone('20121203-0013')
#obs.sb.instruction_set="run-obs-script ~/scripts/observation/spiral_holography_scan.py -f 1722 -b ant5 --scan-extent 6 --cycle-duration 6000 --num-cycles 1 --kind 'uniform' '3C 286' --stow-when-done"
#look on http://kat-flap.control.kat.ac.za/kat/KatGUI.swf and connect to 'karoo from site'

import time

import katpoint
# Import script helper functions from observe.py
try:
    from katcorelib import (standard_script_options, verify_and_connect,collect_targets, start_session, user_logger, ant_array)
    testmode=False
except:
    import optparse
    testmode=True
    standard_script_options=optparse.OptionParser
    
import numpy as np
import scipy
from scikits.fitting import NonLinearLeastSquaresFit, PiecewisePolynomial1DFit
try:
    import matplotlib.pyplot as plt
except:
    pass

#anystowed=np.any([res._returns[0][4]=='STOW' for res in all_ants.req.sensor_value('mode').values()])
def plane_to_sphere_holography(targetaz,targetel,ll,mm):
    scanaz=targetaz-np.arcsin(np.clip(ll/np.cos(targetel),-1.0,1.0))
    scanel=np.arcsin(np.clip((np.sqrt(1.0-ll**2-mm**2)*np.sin(targetel)+np.sqrt(np.cos(targetel)**2-ll**2)*mm)/(1.0-ll**2),-1.0,1.0))
    return scanaz,scanel

#same as katpoint.projection._sphere_to_plane_common(az0=scanaz,el0=scanel,az=targetaz,el=targetel) with ll=ortho_x,mm=-ortho_y
def sphere_to_plane_holography(targetaz,targetel,scanaz,scanel):
    #produces direction cosine coordinates from scanning antenna azimuth,elevation coordinates
    #see _coordinate options.py for derivation
    ll=np.cos(targetel)*np.sin(targetaz-scanaz)
    mm=np.cos(targetel)*np.sin(scanel)*np.cos(targetaz-scanaz)-np.cos(scanel)*np.sin(targetel)
    return ll,mm

def spiral(params,indep):
    x0=indep[0]
    y0=indep[1]
    r=params[0]
    x=r*np.cos(2.0*np.pi*r)
    y=r*np.sin(2.0*np.pi*r)
    return np.sqrt((x-x0)**2+(y-y0)**2)

def SplitArray(x,y,doplot=False):
    #groups antennas into two groups that ensures that shortest possible baselines are used for long baseline antennas
    dist=np.zeros(len(x))
    mindist=np.zeros(len(x))
    nmindist=np.zeros(len(x))
    imindist=[0 for c in range(len(x))]
    for ix in range(len(x)):
        dists=np.sqrt((x[ix]-x)**2+(y[ix]-y)**2)
        dists[ix]=np.nan
        mindist[ix]=np.nanmin(dists)
        imindist[ix]=np.nanargmin(dists)

    if (doplot):
        plt.figure()
        plt.clf()
        for ix in range(len(x)):
            plt.text(x[ix],y[ix],' %d'%(ix))

    oppositegroups=[]
    for cix in range(len(x)):
        ix=np.nanargmax(mindist)
        oppositegroups.append((ix,imindist[ix]))
        mindist[ix]=np.nan
    # print oppositegroups
    GroupA=[oppositegroups[0][0]]
    GroupB=[oppositegroups[0][1]]
    oppositegroups.pop(0)
    while (len(oppositegroups)):
        exited=False
        for io,pair in enumerate(oppositegroups):
            if (pair[0] in GroupA):
                if (pair[1] not in GroupB):
                     GroupB.append(pair[1])
                oppositegroups.pop(io)
                exited=True
                break
            if (pair[1] in GroupA):
                if (pair[0] not in GroupB):
                    GroupB.append(pair[0])
                oppositegroups.pop(io)
                exited=True
                break
            if (pair[0] in GroupB):
                if (pair[1] not in GroupA):
                    GroupA.append(pair[1])
                oppositegroups.pop(io)
                exited=True
                break
            if (pair[1] in GroupB):
                if (pair[0] not in GroupA):
                    GroupA.append(pair[0])
                oppositegroups.pop(io)
                exited=True
                break
        if (exited is False):#so no more absolute exclusions clear at this point
            #is pair[0] closer to closest antenna from group A or group B compared to pair[1]?
            # print oppositegroups
            pair=oppositegroups.pop(0)
            distsA0=np.sqrt((x[GroupA]-x[pair[0]])**2+(y[GroupA]-y[pair[0]])**2)
            distsB0=np.sqrt((x[GroupB]-x[pair[0]])**2+(y[GroupB]-y[pair[0]])**2)
            distsA1=np.sqrt((x[GroupA]-x[pair[1]])**2+(y[GroupA]-y[pair[1]])**2)
            distsB1=np.sqrt((x[GroupB]-x[pair[1]])**2+(y[GroupB]-y[pair[1]])**2)
            # print np.min(distsA0),np.min(distsB1),np.min(distsA1),np.min(distsB0)
            if (np.max([np.min(distsA0),np.min(distsB1)])<np.max([np.min(distsA1),np.min(distsB0)])):
                GroupA.append(pair[0])
                GroupB.append(pair[1])
            else:
                GroupA.append(pair[1])
                GroupB.append(pair[0])

    if (doplot):
        plt.plot(x[GroupA],y[GroupA],'.r')
        plt.plot(x[GroupB],y[GroupB],'.g')
        plt.axis('equal')
    return GroupA,GroupB

#note that we want spiral to only extend to above horizon for first few scans in case source is rising
#should test if source is rising or setting before each composite scan, and use -compositey if setting
#slowtime redistributes samples on each arm so that start and stop of scan occurs slower within this timerange in seconds
def generatespiral(totextent,tottime,tracktime=1,slewtime=1,slowtime=1,sampletime=1,spacetime=1,kind='uniform',mirrorx=False,num_scans=None,scan_duration=None,polish_factor=1.0):
    totextent=np.float(totextent)
    tottime=np.float(tottime)
    sampletime=np.float(sampletime)
    spacetime=np.float(spacetime)
    nextrazeros=int(np.float(tracktime)/sampletime)
    nextraslew=int(np.float(slewtime)/sampletime)
    tracktime=nextrazeros*sampletime
    slewtime=nextraslew*sampletime
    radextent=np.float(totextent)/2.0
    if (kind=='dense-core'):
        c=np.sqrt(2)*180.0/(16.0*np.pi)
        narms=2*int(np.sqrt(tottime/c+((tracktime+slewtime)/c)**2)-(tracktime+slewtime)/c)#ensures even number of arms - then scan pattern ends on target (if odd it will not)
        ntime=int((tottime-(tracktime+slewtime)*narms)/(sampletime*narms))
        armrad=radextent*(np.linspace(0,1,ntime))
        armtheta=np.linspace(0,np.pi,ntime)
        armx=armrad*np.cos(armtheta)
        army=armrad*np.sin(armtheta)
    elif (kind=='approx'):
        c=180.0/(16.0*np.pi)
        narms=2*int(np.sqrt(tottime/c+((tracktime+slewtime)/c)**2)-(tracktime+slewtime)/c)#ensures even number of arms - then scan pattern ends on target (if odd it will not)
        ntime=int((tottime-(tracktime+slewtime)*narms)/(sampletime*narms))
        armrad=radextent*(np.linspace(0,1,ntime))
        armtheta=np.linspace(0,np.pi,ntime)
        armx=armrad*np.cos(armtheta)
        army=armrad*np.sin(armtheta)
        dist=np.sqrt((armx[:-1]-armx[1:])**2+(army[:-1]-army[1:])**2)
        narmrad=np.cumsum(np.concatenate([np.array([0]),1.0/dist]))
        narmrad*=radextent/max(narmrad)
        narmtheta=narmrad/radextent*np.pi
        armx=narmrad*np.cos(narmtheta)
        army=narmrad*np.sin(narmtheta)
    elif (kind=='raster' or kind=='rasterx' or kind=='rastery'):
        if (slewtime<radextent*4.):
            user_logger.info("Warning: adjusting slewtime from %g to %g because value unrealistic for this scan pattern", slewtime, radextent*4)
            slewtime=radextent*4.#antenna can slew at 1 degrees per second in elevation
        narms=num_scans
        ntime=int((tottime/num_scans-tracktime*2-slewtime*2.0)/sampletime)
        armx=np.zeros(ntime)
        army=np.zeros(ntime)
        if (slowtime>0.0):
            repl=np.linspace(0.0,slowtime/sampletime,2+int((slowtime)/sampletime))
            dt=np.float(slowtime)/np.float(sampletime)*np.ones(ntime,dtype='float')
            dt[:len(repl)-1]=repl[1:]
            dt[1-len(repl):]=repl[:0:-1]
            dt=dt/np.sum(dt)
            sdt=np.float(slowtime)/np.float(sampletime)*np.ones(int(slewtime/sampletime),dtype='float')
            sdt[:len(repl)-1]=repl[1:]
            sdt[1-len(repl):]=repl[:0:-1]
            sdt=sdt/np.sum(sdt)
        else:
            dt=1.0/ntime*np.ones(ntime)
            sdt=1.0/(slewtime/sampletime)*np.ones(int(slewtime/sampletime))
        scan=np.cumsum(dt)
        scan=((scan-scan[0])/(scan[-1]-scan[0])-0.5)*radextent*2
        slew=np.cumsum(sdt)
        slew=((slew-slew[0])/(slew[-1]-slew[0]))*radextent
        nextraslew=len(slew)
        fullscanx=np.r_[np.zeros(int(tracktime/sampletime)),-slew,scan,slew[::-1],np.zeros(int(tracktime/sampletime))]
        fullscany=np.r_[np.zeros(int(tracktime/sampletime)),slew/radextent,np.ones(len(scan)),slew[::-1]/radextent,np.zeros(int(tracktime/sampletime))]
        compositex=[[] for ia in range(narms)]
        compositey=[[] for ia in range(narms)]
        ncompositex=[[] for ia in range(narms)]
        ncompositey=[[] for ia in range(narms)]
        if (kind=='rastery'):
            for ia,y in enumerate(np.linspace(-radextent,radextent,num_scans)):
                compositey[ia]=fullscanx
                compositex[ia]=fullscany*y
                ncompositey[ia]=fullscanx
                ncompositex[ia]=fullscany*y
        else:
            for ia,y in enumerate(np.linspace(-radextent,radextent,num_scans)):
                compositex[ia]=fullscanx
                compositey[ia]=-fullscany*y#when target is rising do top first, doing rasterx pattern. For rastery pattern it has no advantage
                ncompositex[ia]=fullscanx
                ncompositey[ia]=fullscany*y
        return compositex,compositey,ncompositex,ncompositey,nextraslew
    elif (kind=='polish'):
        nextraslew=0
        nleaves=int(tottime/(40.*np.sqrt(spacetime)+2*tracktime))
        nptsperarm=(tottime/np.float(nleaves)-2*tracktime)/sampletime #time per circle
        t=np.pi*np.tanh(polish_factor*np.linspace(-1,1,nptsperarm))/np.tanh(polish_factor)
        ix=np.zeros(len(t))
        iy=np.zeros(len(t))
        x=np.sin(t)
        y=(np.cos(t)+1.)
        compositex=[]
        compositey=[]
        ncompositex=[]
        ncompositey=[]
        for theta in np.linspace(-np.pi/2,-np.pi/2+2.*np.pi,nleaves,endpoint=False):
            for i,itheta in enumerate(np.linspace(0,2.*np.pi/nleaves,len(t))):
                ix[i]=x[i]*np.cos(itheta)+(y[i])*np.sin(itheta)
                iy[i]=y[i]*np.cos(itheta)-x[i]*np.sin(itheta)        
            xx=totextent/4.*(ix*np.cos(theta)+(iy)*np.sin(theta))
            yy=totextent/4.*(iy*np.cos(theta)-ix*np.sin(theta))
            compositex.append(np.r_[np.repeat(0.0,nextrazeros),xx,np.repeat(0.0,nextrazeros)])
            compositey.append(np.r_[np.repeat(0.0,nextrazeros),yy,np.repeat(0.0,nextrazeros)])
            ncompositex.append(np.r_[np.repeat(0.0,nextrazeros),xx,np.repeat(0.0,nextrazeros)])
            ncompositey.append(np.r_[np.repeat(0.0,nextrazeros),-yy,np.repeat(0.0,nextrazeros)])
        return compositex,compositey,ncompositex,ncompositey,nextraslew
    elif (kind=='circle'):
        ncircles=int(tottime/(40.*np.sqrt(spacetime)+tracktime))
        ntime=(tottime/np.float(ncircles)-tracktime)/sampletime #time per circle
        compositex=[]
        compositey=[]
        r0=np.linspace(0,2,int(ntime/2))#for 1/r
        #r0=(r0**2)/4.#for 1/r2
        #r0=np.sqrt(r0)*np.sqrt(2.-1e-9)#for uniform
        r1=1.
        x=0.5*(1+r0**2-r1**2)
        y=np.sqrt(r0**2-x**2)
        x=np.r_[np.repeat(0.0,int(np.ceil(tracktime/sampletime/2.))),x,x[-2::-1],np.repeat(0.0,int(np.floor(tracktime/sampletime/2.)))]/4.
        y=np.r_[np.repeat(0.0,int(np.ceil(tracktime/sampletime/2.))),y,-y[-2::-1],np.repeat(0.0,int(np.floor(tracktime/sampletime/2.)))]/4.
        for th in np.linspace(0,360,ncircles,endpoint=False):
            nx=x*np.cos(th*np.pi/180.)-y*np.sin(th*np.pi/180.)
            ny=x*np.sin(th*np.pi/180.)+y*np.cos(th*np.pi/180.)
            compositex.append(nx*totextent)
            compositey.append(ny*totextent)
        return compositex,compositey,compositex,compositey,0
    elif (kind=='radial'):
        c=180.0/(16.0*np.pi)
        narms=2*int(1.5*(np.sqrt(tottime/c+((tracktime+slewtime)/c)**2)-(tracktime+slewtime)/c))#ensures even number of arms - then scan pattern ends on target (if odd it will not)
        print 'narms',narms
        ntime=int((tottime-(tracktime+slewtime)*narms)/(sampletime*narms))
        armx=np.zeros(ntime)
        army=np.zeros(ntime)
        #must be on curve x=t*cos(np.pi*t),y=t*sin(np.pi*t)
        #intersect (x-x0)**2+(y-y0)**2=1/ntime**2 with spiral
        if (slowtime>0.0):
            repl=np.linspace(0.0,slowtime/sampletime,2+(slowtime)/sampletime)
            dt=np.float(slowtime)/np.float(sampletime)*np.ones(ntime,dtype='float')
            dt[:len(repl)-1]=repl[1:]
            dt[1-len(repl):]=repl[:0:-1]
            dt=dt/np.sum(dt)
        else:
            dt=1.0/ntime*np.ones(ntime)
        armx=np.cumsum(dt)
        maxrad=np.sqrt(armx[-1]**2+army[-1]**2)
        armx=armx*radextent/maxrad
    else:#'uniform'
        c=180.0/(16.0*np.pi)
        narms=2*int(np.sqrt(tottime/spacetime/c+((tracktime+slewtime)/c)**2)-(tracktime+slewtime)/c)#ensures even number of arms - then scan pattern ends on target (if odd it will not)
        ntime=int((tottime-(tracktime+slewtime)*narms)/(sampletime*narms))
        armx=np.zeros(ntime)
        army=np.zeros(ntime)
        #must be on curve x=t*cos(np.pi*t),y=t*sin(np.pi*t)
        #intersect (x-x0)**2+(y-y0)**2=1/ntime**2 with spiral
        if (slowtime>0.0):
            repl=np.linspace(0.0,slowtime/sampletime,2+(slowtime)/sampletime)
            dt=np.float(slowtime)/np.float(sampletime)*np.ones(ntime,dtype='float')
            dt[:len(repl)-1]=repl[1:]
            dt[1-len(repl):]=repl[:0:-1]
            dt=dt/np.sum(dt)
        else:
            dt=1.0/ntime*np.ones(ntime)
        lastr=0.0
        for it in range(1,ntime):
            data=np.array([dt[it-1]])
            indep=np.array([armx[it-1],army[it-1]])#last calculated coordinate in arm, is x0,y0
            initialparams=np.array([lastr+dt[it-1]]);
            fitter=NonLinearLeastSquaresFit(spiral,initialparams)
            fitter.fit(indep,data)
            lastr=fitter.params[0];
            armx[it]=lastr*np.cos(2.0*np.pi*lastr)
            army[it]=lastr*np.sin(2.0*np.pi*lastr)

        maxrad=np.sqrt(armx[it]**2+army[it]**2)
        armx=armx*radextent/maxrad
        army=army*radextent/maxrad
    # ndist=sqrt((armx[:-1]-armx[1:])**2+(army[:-1]-army[1:])**2)
    # print ndist

    compositex=[[] for ia in range(narms)]
    compositey=[[] for ia in range(narms)]
    ncompositex=[[] for ia in range(narms)]
    ncompositey=[[] for ia in range(narms)]
    reverse=False
    for ia in range(narms):
        rot=-ia*np.pi*2.0/narms
        x=armx*np.cos(rot)-army*np.sin(rot)
        y=armx*np.sin(rot)+army*np.cos(rot)
        nrot=ia*np.pi*2.0/narms
        nx=armx*np.cos(nrot)-army*np.sin(nrot)
        ny=armx*np.sin(nrot)+army*np.cos(nrot)
        if (nextrazeros>0):
            x=np.r_[np.repeat(0.0,nextrazeros),x]
            y=np.r_[np.repeat(0.0,nextrazeros),y]
            nx=np.r_[np.repeat(0.0,nextrazeros),nx]
            ny=np.r_[np.repeat(0.0,nextrazeros),ny]
        if reverse:
            reverse=False
            x=x[::-1]
            y=y[::-1]
            nx=nx[::-1]
            ny=ny[::-1]
        else:
            reverse=True
        compositex[ia]=x
        compositey[ia]=y
        ncompositex[ia]=nx
        ncompositey[ia]=ny
    if (nextraslew>0):
        for ia in range(0,narms,2):
            inter=np.cumsum(np.r_[np.linspace(0,1,1+nextraslew,endpoint=False),np.linspace(1,0,1+nextraslew,endpoint=False)])
            inter/=inter[-1]
            interx=compositex[ia][-1]+(compositex[ia+1][0]-compositex[ia][-1])*inter
            compositex[ia]=np.r_[compositex[ia],interx[1:1+nextraslew]]
            compositex[ia+1]=np.r_[interx[1+nextraslew:-1],compositex[ia+1]]
            intery=compositey[ia][-1]+(compositey[ia+1][0]-compositey[ia][-1])*inter
            compositey[ia]=np.r_[compositey[ia],intery[1:1+nextraslew]]
            compositey[ia+1]=np.r_[intery[1+nextraslew:-1],compositey[ia+1]]
            ninterx=ncompositex[ia][-1]+(ncompositex[ia+1][0]-ncompositex[ia][-1])*inter
            ncompositex[ia]=np.r_[ncompositex[ia],ninterx[1:1+nextraslew]]
            ncompositex[ia+1]=np.r_[ninterx[1+nextraslew:-1],ncompositex[ia+1]]
            nintery=ncompositey[ia][-1]+(ncompositey[ia+1][0]-ncompositey[ia][-1])*inter
            ncompositey[ia]=np.r_[ncompositey[ia],nintery[1:1+nextraslew]]
            ncompositey[ia+1]=np.r_[nintery[1+nextraslew:-1],ncompositey[ia+1]]
    if (mirrorx):
        for ia in range(narms):
            compositex[ia]=-compositex[ia]
            ncompositex[ia]=-ncompositex[ia]

    return compositex,compositey,ncompositex,ncompositey,nextraslew

#high_elevation_slowdown_factor: normal speed up to 60degrees elevation slowed down linearly by said factor at 90 degrees elevation
#note due to the azimuth branch cut moved to -135 degrees, it gives 45 degrees (center to extreme) azimuth range before hitting limits either side
#for a given 10 degree (5 degree center to extreme) scan, this limits maximum elevation of a target to np.arccos(5./45)*180./np.pi=83.62 degrees before experiencing unachievable azimuth values within a scan arm in the worst case scenario
#note scan arms are unwrapped based on current target azimuth position, so may choose new branch cut for next scan arm, possibly corrupting current cycle.
def gen_scan(lasttime,target,az_arm,el_arm,timeperstep,high_elevation_slowdown_factor=1.0,clip_safety_margin=1.0,min_elevation=15.,max_elevation=90.):
    num_points = np.shape(az_arm)[0]
    az_arm = az_arm*np.pi/180.0
    el_arm = el_arm*np.pi/180.0
    scan_data = np.zeros((num_points,3))
    attime = lasttime+np.arange(1,num_points+1)*timeperstep
    #spiral arm scan
    targetaz_rad,targetel_rad=target.azel(attime)#gives targetaz in range 0 to 2*pi
    targetaz_rad=((targetaz_rad+135*np.pi/180.)%(2.*np.pi)-135.*np.pi/180.)#valid steerable az is from -180 to 270 degrees so move branch cut to -135 or 225 degrees
    scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,az_arm ,el_arm)
    if high_elevation_slowdown_factor>1.0:
        meanscanarmel=np.mean(scanel)*180./np.pi
        if meanscanarmel>60.:#recompute slower scan arm based on average elevation at if measured at normal speed
            slowdown_factor=(meanscanarmel-60.)/(90.-60.)*(high_elevation_slowdown_factor-1.)+1.0#scales linearly from 1 at 60 deg el, to high_elevation_slowdown_factor at 90 deg el
            attime = lasttime+np.arange(1,num_points+1)*timeperstep*slowdown_factor
            targetaz_rad,targetel_rad=target.azel(attime)#gives targetaz in range 0 to 2*pi
            targetaz_rad=((targetaz_rad+135*np.pi/180.)%(2.*np.pi)-135.*np.pi/180.)#valid steerable az is from -180 to 270 degrees so move branch cut to -135 or 225 degrees
            scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,az_arm ,el_arm)
    #clipping prevents antenna from hitting hard limit and getting stuck until technician reset it, 
    #but not ideal to reach this because then actual azel could be equal to requested azel even though not right and may falsely cause one to believe everything is ok
    azdata=np.unwrap(scanaz)*180.0/np.pi
    eldata=scanel*180.0/np.pi
    scan_data[:,0] = attime
    scan_data[:,1] = np.clip(np.nan_to_num(azdata),-180.0+clip_safety_margin,270.0-clip_safety_margin)
    scan_data[:,2] = np.clip(np.nan_to_num(eldata),min_elevation+clip_safety_margin,max_elevation-clip_safety_margin)
    clipping_occurred=(np.sum(azdata==scan_data[:,1])+np.sum(eldata==scan_data[:,2])!=len(eldata)*2)
    return scan_data,clipping_occurred

def gen_track(attime,target):
    track_data = np.zeros((len(attime),3))
    targetaz_rad,targetel_rad=target.azel(attime)#gives targetaz in range 0 to 2*pi
    targetaz_rad=((targetaz_rad+135*np.pi/180.)%(2.*np.pi)-135.*np.pi/180.)#valid steerable az is from -180 to 270 degrees so move branch cut to -135 or 225 degrees
    track_data[:,0] = attime
    track_data[:,1] = np.unwrap(targetaz_rad)*180.0/np.pi
    track_data[:,2] = targetel_rad*180.0/np.pi
    return track_data

def test_target_azel_limits(target,clip_safety_margin,min_elevation,max_elevation):
    now=time.time()
    targetazel=gen_track([now],target)[0][1:]
    slewtotargettime=np.max([0.5*np.abs(currentaz-targetazel[0]),1.*np.abs(currentel-targetazel[1])])+1.0#antenna can slew at 2 degrees per sec in azimuth and 1 degree per sec in elev
    starttime=now+slewtotargettime+opts.cycle_tracktime
    targetel=np.array(target.azel([starttime,starttime+1.])[1])*180.0/np.pi
    rising=targetel[1]>targetel[0]
    if rising:#target is rising - scan top half of pattern first
        cx=compositex
        cy=compositey
    else:  #target is setting - scan bottom half of pattern first
        cx=ncompositex
        cy=ncompositey
    for iarm in range(len(cx)):#spiral arm index
        scan_data,clipping_occurred = gen_scan(starttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=clip_safety_margin,min_elevation=min_elevation,max_elevation=max_elevation)
        starttime=scan_data[-1,0]
        if clipping_occurred:
            return False, rising, starttime-now
    return True, rising, starttime-now

if __name__=="__main__":
    # Set up standard script options
    parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                     description='This script performs a holography scan on the specified target. '
                                                 'All the antennas initially track the target, whereafter a subset '
                                                 'of the antennas (the "scan antennas" specified by the --scan-ants '
                                                 'option) perform a scan on the target. Note also some '
                                                 '**required** options below. Targets ordered by preference.')
    # Add experiment-specific options
    parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna). Could also be GroupA or GroupB to select half of available antennas automatically. GroupAB to alternate between GroupA and GroupB. An integer specifies number of scanning antennas, chosen automatically.',default='GroupAB')
    parser.add_option('--track-ants', help='Subset of all antennas that will track source. An integer specifies number of tracking antennas, chosen automatically. (default=all non-scanning antennas)')
    parser.add_option('--num-cycles', type='int', default=-1,
                      help='Number of beam measurement cycles to complete (default=%default) use -1 for indefinite')
    parser.add_option('--num-scans', type='int', default=None,
                      help='Number of raster scans or spiral arms in scan pattern. This value is usually automatically determined. (default=%default)')
    parser.add_option('--scan-duration', type='float', default=None,
                      help='Time in seconds to spend on each scan. This value is usually automatically determined (default=%default)')
    parser.add_option('--cycle-duration', type='float', default=1800,
                      help='Time to spend measuring beam pattern per cycle, in seconds (default=%default)')
    parser.add_option('-l', '--scan-extent', type='float', default=10,
                      help='Diameter of beam pattern to measure, in degrees (default=%default)')
    parser.add_option('--kind', type='string', default='uniform',
                      help='Kind of spiral, could be "radial", "raster", "uniform" or "dense-core" (default=%default)')
    parser.add_option('--tracktime', type='float', default=2,
                      help='Extra time in seconds for scanning antennas to track when passing over target (default=%default)')
    parser.add_option('--cycle-tracktime', type='float', default=30,
                      help='Extra time in seconds for scanning antennas to track when passing over target (default=%default)')
    parser.add_option('--slewtime', type='float', default=3,
                      help='Extra time in seconds for scanning antennas to slew when passing from one spiral arm to another (default=%default)')
    parser.add_option('--slowtime', type='float', default=6,
                      help='Time in seconds to slow down at start and end of each spiral arm (default=%default)')
    parser.add_option('--sampletime', type='float', default=0.25,
                      help='time in seconds to spend on pointing (default=%default)')
    parser.add_option('--spacetime', type='float', default=3,
                      help='time in seconds used to equalize arm spacing, match with dumprate for equal two-dimensional sample spacing (default=%default)')
    parser.add_option('--polish-factor', type='float', default=1.0,
                      help='factor by which to slow down nominal scanning close to boresight for polish scan pattern (default=%default)')
    parser.add_option('--high-elevation-slowdown-factor', type='float', default=2.0,
                      help='factor by which to slow down nominal scanning speed at 90 degree elevation, linearly scaled from factor of 1 at 60 degrees elevation (default=%default)')
    parser.add_option('--target-elevation-override', type='float', default=90.0,
                      help='Honour preferred target order except if lower ranking target exceeds this elevation limit (default=%default). Use this feature to capture high elevation targets when available.')
    parser.add_option('--target-low-elevation-override', type='float', default=0.0,
                      help='Honour preferred target order except if lower ranking target below this elevation limit (default=%default). Use this feature to capture low elevation targets when available.')
    parser.add_option('--prepopulatetime', type='float', default=10.0,
                      help='time in seconds to prepopulate buffer in advance (default=%default)')
    parser.add_option('--mirrorx', action="store_true", default=False,
                      help='Mirrors x coordinates of pattern (default=%default)')
    parser.add_option('--auto-delay', type='string', default=None,
                      help='Set CBF auto-delay on or off (default=%default)')
    parser.add_option('--debugtrack', action="store_true", default=False,
                      help='disables load_scan tracking command (default=%default)')
                  
    # Set default value for any option (both standard and experiment-specific options)
    parser.set_defaults(description='Spiral holography scan', quorum=1.0, nd_params='off')
    # Parse the command line
    opts, args = parser.parse_args()

    compositex,compositey,ncompositex,ncompositey,nextraslew=generatespiral(totextent=opts.scan_extent,tottime=opts.cycle_duration,tracktime=opts.tracktime,slewtime=opts.slewtime,slowtime=opts.slowtime,sampletime=opts.sampletime,spacetime=opts.spacetime,kind=opts.kind,mirrorx=opts.mirrorx,num_scans=opts.num_scans,scan_duration=opts.scan_duration,polish_factor=opts.polish_factor)
    if testmode:
        plt.figure()
        x=[]
        y=[]
        for iarm in range(len(compositex)):
            plt.plot(compositex[iarm],compositey[iarm],'.')
            x.extend(compositex[iarm])
            y.extend(compositey[iarm])
        plt.figure()
        plt.subplot(3,1,1)
        t=np.arange(len(x))*np.float(opts.sampletime)
        plt.plot(t,x,'-')
        plt.plot(t,y,'--')
        plt.ylabel('[degrees]')
        plt.legend(['x','y'])
        plt.title('Position profile')
        plt.subplot(3,1,2)
        t=np.arange(len(x))*np.float(opts.sampletime)
        plt.plot(t[:-1],np.diff(x)/opts.sampletime,'-')
        plt.plot(t[:-1],np.diff(y)/opts.sampletime,'--')
        plt.ylabel('[degrees/s]')
        plt.legend(['dx','dy'])
        plt.title('Speed profile')
        plt.subplot(3,1,3)
        t=np.arange(len(x))*np.float(opts.sampletime)
        plt.plot(t[:-2],np.diff(np.diff(x))/opts.sampletime/opts.sampletime,'-')
        plt.plot(t[:-2],np.diff(np.diff(y))/opts.sampletime/opts.sampletime,'--')
        plt.ylabel('[degrees/s^2]')
        plt.legend(['ddx','ddy'])
        plt.title('Acceleration profile')
        plt.show()
    else:
        if len(args) == 0:
            args=['3C 273','PKS 1934-63','3C 279','PKS 0408-65','PKS 0023-26','J0825-5010','PKS J1924-2914','Hyd A']

        if 'J0825-5010' in args:#not in catalogue
            args[args.index('J0825-5010')]='J0825-5010,radec, 08:25:26.869, -50:10:38.4877'

        # Check basic command-line options and obtain a kat object connected to the appropriate system
        with verify_and_connect(opts) as kat:
            catalogue = collect_targets(kat, args)
            targets=catalogue.targets
            if len(targets) == 0:
                raise ValueError("Please specify a target argument via name ('Ori A'), "
                                 "description ('azel, 20, 30') or catalogue file name ('sources.csv')")
            # Initialise a capturing session
            with start_session(kat, **vars(opts)) as session:
                # Use the command-line options to set up the system
                session.standard_setup(**vars(opts))
                #determine scan antennas
                all_ants = session.ants
                session.obs_params['num_scans'] = len(compositex)
                grouprange = [0]
                if (opts.track_ants and opts.track_ants.isdigit()):
                    GroupA,GroupB=SplitArray(np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[0] for ant in session.ants]),np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[1] for ant in session.ants]),doplot=False)
                    GroupA.extend(GroupB[::-1])
                    GroupA=GroupA[:-int(opts.track_ants)]
                    scan_ants = ant_array(kat, [session.ants[ant] for ant in GroupA], 'scan_ants')
                elif (opts.track_ants):
                    track_ants = ant_array(kat, opts.track_ants, 'track_ants')
                    scan_ants = [ant for ant in all_ants if ant not in track_ants]
                    scan_ants = ant_array(kat, scan_ants, 'scan_ants')
                elif (opts.scan_ants.isdigit()):
                    GroupA,GroupB=SplitArray(np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[0] for ant in session.ants]),np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[1] for ant in session.ants]),doplot=False)
                    GroupA.extend(GroupB[::-1])
                    GroupA=GroupA[:int(opts.scan_ants)]
                    scan_ants = ant_array(kat, [session.ants[ant] for ant in GroupA], 'scan_ants')
                elif (opts.scan_ants.lower()=='groupa' or opts.scan_ants.lower()=='groupab' or opts.scan_ants.lower()=='groupb'):
                    GroupA,GroupB=SplitArray(np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[0] for ant in session.ants]),np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[1] for ant in session.ants]),doplot=False)
                    scan_ants = ant_array(kat, [session.ants[ant] for ant in (GroupA if (opts.scan_ants.lower()=='groupa' or opts.scan_ants.lower()=='groupab') else GroupB)], 'scan_ants')
                    if (opts.scan_ants.lower()=='groupab'):
                        grouprange = range(2)
                else:
                    # Form scanning antenna subarray (or pick the first antenna as the default scanning antenna)
                    scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else session.ants[0], 'scan_ants')

                # Assign rest of antennas to tracking antenna subarray (or use given antennas)
                track_ants = [ant for ant in all_ants if ant not in scan_ants]
                track_ants = ant_array(kat, track_ants, 'track_ants')
                track_ants_array = [ant_array(kat, [track_ant], 'track_ant') for track_ant in track_ants]
                scan_ants_array = [ant_array(kat, [scan_ant], 'scan_ant') for scan_ant in scan_ants]

                # Add metadata
                #note obs_params is immutable and can only be changed before capture_start is called
                session.obs_params['scan_ants']=','.join(np.sort([ant.name for ant in scan_ants]))
                session.obs_params['track_ants']=','.join(np.sort([ant.name for ant in track_ants]))
                # Get observers
                scan_observers = [katpoint.Antenna(scan_ant.sensor.observer.get_value()) for scan_ant in scan_ants]
                track_observers = [katpoint.Antenna(track_ant.sensor.observer.get_value()) for track_ant in track_ants]
                # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
                nd_params = session.nd_params
                session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
                # This also does capture_init, which adds capture_block_id view to telstate and saves obs_params
                session.capture_start()
                session.telstate.add('obs_label','cycle.group.scan')

                user_logger.info("Initiating spiral holography scan cycles (%d %g-second "
                                 "cycles extending %g degrees) on targets %s",
                                 opts.num_cycles, opts.cycle_duration,
                                 opts.scan_extent, ','.join(["'%s'"%(t.name) for t in targets]))

                lasttime = time.time()
                cycle=0
                while cycle<opts.num_cycles or opts.num_cycles<0:
                    if opts.num_cycles<0:
                        user_logger.info("Performing scan cycle %d of unlimited", cycle + 1)
                    else:
                        user_logger.info("Performing scan cycle %d of %d", cycle + 1, opts.num_cycles)
                    user_logger.info("Using all antennas: %s",' '.join([ant.name for ant in session.ants]))

                    for igroup in grouprange:
                        #determine current azel for all antennas
                        currentaz=np.zeros(len(all_ants))
                        currentel=np.zeros(len(all_ants))
                        for iant,ant in enumerate(all_ants):
                            currentaz[iant]=ant.sensor.pos_actual_scan_azim.get_value()
                            currentel[iant]=ant.sensor.pos_actual_scan_elev.get_value()
                        #choose target
                        target=None
                        rising=False
                        expected_duration=None
                        if target is None:#find high elevation target if available
                            for overridetarget in targets:#choose override lower priority target if its minimum elevation is higher than opts.target_elevation_override
                                suitable, rising, expected_duration = test_target_azel_limits(overridetarget,clip_safety_margin=2.0,min_elevation=opts.target_elevation_override,max_elevation=90.)
                                if suitable:
                                    target=overridetarget
                                    break
                        if target is None:#find low elevation target if available
                            for overridetarget in targets:#choose override lower priority target if its minimum elevation is higher than opts.target_elevation_override
                                suitable, rising, expected_duration = test_target_azel_limits(overridetarget,clip_safety_margin=2.0,min_elevation=opts.horizon,max_elevation=opts.target_low_elevation_override)
                                if suitable:
                                    target=overridetarget
                                    break
                        if target is None:#no override found, normal condition
                            for testtarget in targets:
                                suitable, rising, expected_duration = test_target_azel_limits(testtarget,clip_safety_margin=2.0,min_elevation=opts.horizon,max_elevation=90.)
                                if suitable:
                                    target=testtarget
                                    break
                        if target is None:
                            user_logger.info("Quitting because none of the preferred targets are up")
                            break
                        else:
                            user_logger.info("Using target '%s'",target.name)
                            user_logger.info("Current scan estimated to complete at UT %s (in %.1f minutes)",time.ctime(time.time()+expected_duration+time.timezone),expected_duration/60.)
                
                        session.set_target(target)
                        user_logger.info("Performing azimuth unwrap")#ensures wrap of session.track is same as being used in load_scan
                        targetazel=gen_track([time.time()+opts.tracktime],target)[0][1:]
                        azeltarget=katpoint.Target('azimuthunwrap,azel,%s,%s'%(targetazel[0], targetazel[1]))
                        session.track(azeltarget, duration=0, announce=False)#azel target

                        user_logger.info("Performing initial track")
                        session.telstate.add('obs_label','track')
                        session.track(target, duration=opts.cycle_tracktime, announce=False)#radec target
                        if opts.auto_delay is not None:
                            user_logger.info("Setting auto delay to "+opts.auto_delay)
                            session.cbf.req.auto_delay(opts.auto_delay)
                            user_logger.info("Performing follow up track")
                            session.telstate.add('obs_label','delay set track')
                            session.track(target, duration=opts.cycle_tracktime, announce=False)
                        if (rising):#target is rising - scan top half of pattern first
                            cx=compositex
                            cy=compositey
                        else:  #target is setting - scan bottom half of pattern first
                            cx=ncompositex
                            cy=ncompositey
                        user_logger.info("Using Track antennas: %s",' '.join([ant.name for ant in track_ants]))
                        lasttime = time.time()
                        for iarm in range(len(cx)):#spiral arm index
                            user_logger.info("Performing scan arm %d of %d.", iarm + 1, len(cx))
                            if (opts.debugtrack):#original
                                session.ants = scan_ants
                                target.antenna = scan_observers[0]
                                scan_data, clipping_occurred = gen_scan(lasttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                                user_logger.info("Using Scan antennas: %s",
                                                 ' '.join([ant.name for ant in session.ants]))
                                if not kat.dry_run:
                                    session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                                session.ants = track_ants
                                target.antenna = track_observers[0]
                                scan_track = gen_track(scan_data[:,0],target)
                                user_logger.info("Using Track antennas: %s",
                                                 ' '.join([ant.name for ant in session.ants]))
                                if not kat.dry_run:
                                    session.load_scan(scan_track[:,0],scan_track[:,1],scan_track[:,2])
                            else:#fix individual target.antenna issue
                                user_logger.info("Using Scan antennas: %s",
                                                 ' '.join([ant.name for ant in scan_ants]))
                                for iant,scan_ant in enumerate(scan_ants):
                                    session.ants = scan_ants_array[iant]
                                    target.antenna = scan_observers[iant]
                                    scan_data, clipping_occurred = gen_scan(lasttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                                    if not kat.dry_run:
                                        if clipping_occurred:
                                            user_logger.info("Warning unexpected clipping occurred in scan pattern")
                                        session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                        
                            if opts.kind[:6]=='raster':
                                nextrazeros=int(np.float(opts.tracktime)/opts.sampletime)
                                session.telstate.add('obs_label','%d.%d.%d'%(cycle,igroup,iarm),ts=scan_data[0,0])
                                session.telstate.add('obs_label','slew',ts=scan_data[nextrazeros,0])
                                session.telstate.add('obs_label','%d.%d.%d'%(cycle,igroup,iarm),ts=scan_data[nextrazeros+nextraslew,0])
                                session.telstate.add('obs_label','slew',ts=scan_data[-(nextrazeros+nextraslew),0])
                                session.telstate.add('obs_label','%d.%d.%d'%(cycle,igroup,iarm),ts=scan_data[-nextrazeros,0])
                            else:
                                if (iarm%2==0):#outward arm
                                    session.telstate.add('obs_label','%d.%d.%d'%(cycle,igroup,iarm),ts=scan_data[0,0])
                                    if (nextraslew>0):
                                        session.telstate.add('obs_label','slew',ts=scan_data[-nextraslew,0])
                                else:#inward arm
                                    session.telstate.add('obs_label','%d.%d.%d'%(cycle,igroup,iarm),ts=scan_data[nextraslew,0])
                            time.sleep(scan_data[-1,0]-time.time()-opts.prepopulatetime)
                            lasttime = scan_data[-1,0]
                        if (len(grouprange)==2):#swap scanning and tracking antennas
                            track_ants,scan_ants=scan_ants,track_ants
                            track_observers,scan_observers=scan_observers,track_observers
                            track_ants_array,scan_ants_array=scan_ants_array,track_ants_array

                        session.telstate.add('obs_label','slew',ts=lasttime)
                        time.sleep(lasttime-time.time())#wait until last coordinate's time value elapsed
                        #set session antennas to all so that stow-when-done option will stow all used antennas and not just the scanning antennas
                        session.ants = all_ants
                        user_logger.info("Safe to interrupt script now if necessary")
                        if kat.dry_run:#only test one group - dryrun takes too long and causes CAM to bomb out
                            user_logger.info("Testing only one group for dry-run")
                            break
                    if kat.dry_run:#only test one cycle - dryrun takes too long and causes CAM to bomb out
                        user_logger.info("Testing only cycle for dry-run")
                        break
                    cycle+=1
