#!/usr/bin/env python
# holography_scan.py supercedes spiral_holography_scan.py, which is kept in case old style patterns needs to be generated.
# Perform spiral/radial/raster holography scan on specified target(s). Used for beam pattern measurements. 
# Uses bezier path interpolation to moderate slew behavior.
# email: mattieu@ska.ac.za

import time

import katpoint
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
    twistfactor=indep[2]
    r=params[0]
    x=r*np.cos(2.0*np.pi*r*twistfactor)
    y=r*np.sin(2.0*np.pi*r*twistfactor)
    return np.sqrt((x-x0)**2+(y-y0)**2)

#velocity vector changes smoothly from vstart to vend from start to end in length and orientation as fn of time
#can add extra acceleration with zero extra velocity at start and end
# v=v0*(t-t0)/(t1-t0)+v1*(1.-(t-t0)/(t1-t0))+K*((t-t0)/(t1-t0))*((t-t0)/(t1-t0)-1.)
# then find best K while integrating to reach end from start
#x0,y0 is first start point
#x1,y1 is second start point (and approxequals nx[0],ny[0] too)
#inbetween is interpolated points
#x2,y2 is first end point (and approx equals nx[-2],ny[-2])
#x3,y3 is second end point (and approx equals nx[-1],ny[-1])
def bezierpath(params,indep):
    Kx,Ky=params
    if len(indep)==8:
        x0,y0,x1,y1,x2,y2,x3,y3=indep
        n=6+2+2*np.sqrt((x1-x2)**2+(y1-y2)**2)/(np.sqrt((x1-x0)**2+(y1-y0)**2)+np.sqrt((x3-x2)**2+(y3-y2)**2))
    else:
        x0,y0,x1,y1,x2,y2,x3,y3,n=indep
        
    t=np.linspace(0,1,int(abs(n)))
    vx=(x1-x0)*(1.-t)+(x3-x2)*(t)+Kx*t*(t-1.)
    vy=(y1-y0)*(1.-t)+(y3-y2)*(t)+Ky*t*(t-1.)
    nx=np.cumsum(vx)+x0
    ny=np.cumsum(vy)+y0
    return nx,ny

def bezierpathcost(params,indep):
    x0,y0,x1,y1,x2,y2,x3,y3=indep[:8]
    nx,ny=bezierpath(params,indep)
    return [(nx[0]-x1),(ny[0]-y1),(nx[-2]-x2),(ny[-2]-y2),(nx[-1]-x3),(ny[-1]-y3)]

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

#may be useful to track antennas scan ability - position error as function of speed, acceleration
#2D histogram of position error vs speed?

#all quantities in seconds except
#kind='spiral','radial','raster','rastery'
#totextent in degrees
#scanspeed,slewspeed is in degrees/second
#sampletime is in seconds per sample
#trackinterval: in number of scans: returns to track for tracktime after this many scans, track for tracktime include scan in front and end of pattern regardless
def generatepattern(totextent=10,tottime=1800,tracktime=5,slowtime=6,sampletime=1,scanspeed=0.15,slewspeed=-1,twistfactor=1.,trackinterval=1,kind='spiral'):
    if slewspeed<0.:#then factor of scanspeed
        slewspeed*=-scanspeed
    radextent=totextent/2.
    if kind=='radial':
        kind='spiral'
        twistfactor=0.
    if kind=='spiral':
        ntime=int(tottime)
        armx=np.zeros(ntime)
        army=np.zeros(ntime)
        #must be on curve x=t*cos(np.pi*t),y=t*sin(np.pi*t)
        #intersect (x-x0)**2+(y-y0)**2=1/ntime**2 with spiral
        dt=np.repeat(scanspeed*sampletime,ntime)#note dt is the distance in degrees between samples
        if (slowtime>0.0):
            repl=np.linspace(0.0,scanspeed*sampletime,2+int(slowtime/sampletime))
            dt[:len(repl)-1]=repl[1:]
        lastr=0.0
        twistfactore=np.float64(twistfactor)/(totextent)
        for it in range(1,ntime):
            data=np.array([dt[it-1]])
            indep=np.array([armx[it-1],army[it-1],twistfactore])#last calculated coordinate in arm, is x0,y0
            initialparams=np.array([lastr+dt[it-1]]);
            fitter=NonLinearLeastSquaresFit(spiral,initialparams)
            fitter.fit(indep,data)
            lastr=fitter.params[0];
            armx[it]=lastr*np.cos(2.0*np.pi*lastr*twistfactore)
            army[it]=lastr*np.sin(2.0*np.pi*lastr*twistfactore)
            if lastr>=radextent:
                break
        #points used for outward arm including start one at origin = it-1 (so armx[it] army[it] is first point out of bounds):
        # outarmx=armx[:it]
        # outarmy=army[:it]
        #rotate such that first arm orientated at 9 O'clock position (and scans go clockwise)
        theta=np.pi+np.arctan2(army[it-1],armx[it-1])
        outarmx=armx[:it]*np.cos(theta)+army[:it]*np.sin(theta)
        outarmy=army[:it]*np.cos(theta)-armx[:it]*np.sin(theta)

        minscantime=len(outarmx)*sampletime*2#'interruptable' per scanpair, not per scan
        #maxnarms=int((tottime)/(minscantime))#if no time spent on slews nor tracktime
        #solve for maxnarms if no time spent on slews; tracktime every trackinterval arm including after last one
        #maxnarms=int((float(tottime)-float(tracktime)*(float(maxnarms)/float(trackinterval)+1.))/float(minscantime))
        perimetertime=2*np.pi*radextent/(scanspeed)#time to scan around perimeter (is done during slew in sections)
        narms=int((tottime-perimetertime-tracktime)/(minscantime+tracktime/(trackinterval)))#if no time spent on slews
        nslew=int(perimetertime/(narms*sampletime))

        theta=np.pi/narms
        inarmx=outarmx[::-1]*np.cos(theta)+outarmy[::-1]*np.sin(theta)
        inarmy=outarmy[::-1]*np.cos(theta)-outarmx[::-1]*np.sin(theta)
        
        indep=[outarmx[-2],outarmy[-2],outarmx[-1],outarmy[-1],inarmx[0],inarmy[0],inarmx[1],inarmy[1],nslew+3]
        fitter=NonLinearLeastSquaresFit(bezierpathcost,[0.,0.])
        fitter.fit(indep,np.zeros(6))
        params=fitter.params
        nx,ny=bezierpath(params,indep)
        slewx,slewy=nx[1:-2],ny[1:-2]

        compositex=[]
        compositey=[]
        compositeslew=[]
        flatx=[]
        flaty=[]
        flatslew=[]
        for arm in range(narms):
            theta=2.*np.pi*arm/narms
            tmpx=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outarmx,slewx,inarmx,np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpy=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outarmy,slewy,inarmy,np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpslew=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),np.zeros(len(outarmy)),np.ones(len(slewy)),np.zeros(len(inarmy)),np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            compositex.append(tmpx*np.cos(theta)+tmpy*np.sin(theta))
            compositey.append(tmpy*np.cos(theta)-tmpx*np.sin(theta))
            compositeslew.append(tmpslew)
            flatx.extend(tmpx*np.cos(theta)+tmpy*np.sin(theta))
            flaty.extend(tmpy*np.cos(theta)-tmpx*np.sin(theta))
            flatslew.extend(tmpslew)
            
    elif kind=='raster' or kind=='rasterx' or kind=='rastery':
        inarmx=np.arange(-radextent,radextent+1e-10,scanspeed*sampletime)
        outarmx=inarmx[::-1]
        meanscantime=len(outarmx)*sampletime#'interruptable' per scan, not scan pair
        perimetertime=2*totextent/scanspeed
        avgslewtime=2*np.sqrt(radextent**2+(radextent/2)**2)/scanspeed#double time for in and out required
        narms=int((tottime-perimetertime*(trackinterval-1)/trackinterval-tracktime)/(meanscantime+tracktime/trackinterval+avgslewtime/trackinterval))
        if narms%2==0:#only allows odd number of scans for raster
            narms-=1
        
        compositex=[]
        compositey=[]
        compositeslew=[]
        flatx=[]
        flaty=[]
        flatslew=[]
        for arm in range(narms):
            y = ((narms-1.)/2.-arm)/float(narms-1.)*totextent
            thisarmx=outarmx[:]
            thisarmy=np.ones(len(thisarmx))*y
            if (arm%2):#alternates direction with every scan
                thisarmx=thisarmx[::-1]
            if (arm%trackinterval==0):
                lastarmx=np.zeros(2)
                lastarmy=np.zeros(2)
            else:
                lasty = ((narms-1.)/2.-(arm-1))/float(narms-1.)*totextent
                lastarmx=outarmx[:]
                lastarmy=np.ones(len(thisarmx))*lasty
                if (arm-1)%2:
                    lastarmx=lastarmx[::-1]
            
            if ((arm+1)%trackinterval==0) or arm==narms-1:
                nextarmx=np.zeros(2)
                nextarmy=np.zeros(2)
            else:
                nexty = ((narms-1.)/2.-(arm+1))/float(narms-1.)*totextent
                nextarmx=outarmx[:]
                nextarmy=np.ones(len(thisarmx))*nexty
                if (arm+1)%2:
                    nextarmx=nextarmx[::-1]
            nslew=np.sqrt((lastarmx[-1]-thisarmx[0])**2+(lastarmy[-1]-thisarmy[0])**2)/(slewspeed*sampletime)
            indep=[lastarmx[-2],lastarmy[-2],lastarmx[-1],lastarmy[-1],thisarmx[0],thisarmy[0],thisarmx[1],thisarmy[1],nslew+3]
            fitter=NonLinearLeastSquaresFit(bezierpathcost,[0.,0.])
            fitter.fit(indep,np.zeros(6))
            params=fitter.params
            nx,ny=bezierpath(params,indep)
            outslewx,outslewy=nx[1:-2],ny[1:-2]

            nslew=np.sqrt((thisarmx[-1]-nextarmx[0])**2+(thisarmy[-1]-nextarmy[0])**2)/(slewspeed*sampletime)
            indep=[thisarmx[-2],thisarmy[-2],thisarmx[-1],thisarmy[-1],nextarmx[0],nextarmy[0],nextarmx[1],nextarmy[1],nslew+3]
            fitter=NonLinearLeastSquaresFit(bezierpathcost,[0.,0.])
            fitter.fit(indep,np.zeros(6))
            params=fitter.params
            nx,ny=bezierpath(params,indep)
            inslewx,inslewy=nx[1:-2],ny[1:-2]

            tmpx=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outslewx,thisarmx,inslewx if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpy=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outslewy,thisarmy,inslewy if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpslew=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),np.ones(len(outslewy)),np.zeros(len(thisarmy)),np.ones(len(inslewy)) if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            

            compositex.append(tmpy if kind=='rastery' else tmpx)
            compositey.append(tmpx if kind=='rastery' else tmpy)
            compositeslew.append(tmpslew)
            flatx.extend(tmpy if kind=='rastery' else tmpx)
            flaty.extend(tmpx if kind=='rastery' else tmpy)
            flatslew.extend(tmpslew)
    elif kind=='rasterdisc':#disc footprint
        inarmx=np.arange(-radextent,radextent+1e-10,scanspeed*sampletime)
        outarmx=inarmx[::-1]
        meanscanlen=(np.pi/2.)/2.*len(outarmx)#mean horizontal scan length when clipped to disc 'interruptable' per scan, not scan pair
        meanscantime=meanscanlen*sampletime
        perimetertime=np.pi*radextent/(scanspeed)#time to scan around perimeter (is done during slew in sections)
        avgslewtime=2*radextent/scanspeed#double time for in and out required
        narms=int((tottime-perimetertime*(trackinterval-1)/trackinterval-tracktime)/(meanscantime+tracktime/trackinterval+avgslewtime/trackinterval))
        if narms%2==0:#only allows odd number of scans for raster
            narms-=1
        
        compositex=[]
        compositey=[]
        compositeslew=[]
        flatx=[]
        flaty=[]
        flatslew=[]
        for arm in range(narms):
            y = ((narms-1.)/2.-arm)/float(narms)*totextent#note dividing here by narms instead of narms-1 so that half spacing away from maximum at top and bottom
            thisarmx=outarmx[:]
            thisarmy=np.ones(len(thisarmx))*y
            valid=np.nonzero(np.sqrt(thisarmx**2+thisarmy**2)<=radextent)[0]
            thisarmx=thisarmx[valid]
            thisarmy=thisarmy[valid]
            if (arm%2):#alternates direction with every scan
                thisarmx=thisarmx[::-1]
            if (arm%trackinterval==0):
                lastarmx=np.zeros(2)
                lastarmy=np.zeros(2)
            else:
                lasty = ((narms-1.)/2.-(arm-1))/float(narms)*totextent
                lastarmx=outarmx[:]
                lastarmy=np.ones(len(outarmx))*lasty
                valid=np.nonzero(np.sqrt(lastarmx**2+lastarmy**2)<=radextent)[0]
                lastarmx=lastarmx[valid]
                lastarmy=lastarmy[valid]
                if (arm-1)%2:
                    lastarmx=lastarmx[::-1]
            
            if ((arm+1)%trackinterval==0) or arm==narms-1:
                nextarmx=np.zeros(2)
                nextarmy=np.zeros(2)
            else:
                nexty = ((narms-1.)/2.-(arm+1))/float(narms)*totextent
                nextarmx=outarmx[:]
                nextarmy=np.ones(len(outarmx))*nexty
                valid=np.nonzero(np.sqrt(nextarmx**2+nextarmy**2)<=radextent)[0]
                nextarmx=nextarmx[valid]
                nextarmy=nextarmy[valid]
                if (arm+1)%2:
                    nextarmx=nextarmx[::-1]
            nslew=np.sqrt((lastarmx[-1]-thisarmx[0])**2+(lastarmy[-1]-thisarmy[0])**2)/(slewspeed*sampletime)
            indep=[lastarmx[-2],lastarmy[-2],lastarmx[-1],lastarmy[-1],thisarmx[0],thisarmy[0],thisarmx[1],thisarmy[1],nslew+3]
            fitter=NonLinearLeastSquaresFit(bezierpathcost,[0.,0.])
            fitter.fit(indep,np.zeros(6))
            params=fitter.params
            nx,ny=bezierpath(params,indep)
            outslewx,outslewy=nx[1:-2],ny[1:-2]

            nslew=np.sqrt((thisarmx[-1]-nextarmx[0])**2+(thisarmy[-1]-nextarmy[0])**2)/(slewspeed*sampletime)
            indep=[thisarmx[-2],thisarmy[-2],thisarmx[-1],thisarmy[-1],nextarmx[0],nextarmy[0],nextarmx[1],nextarmy[1],nslew+3]
            fitter=NonLinearLeastSquaresFit(bezierpathcost,[0.,0.])
            fitter.fit(indep,np.zeros(6))
            params=fitter.params
            nx,ny=bezierpath(params,indep)
            inslewx,inslewy=nx[1:-2],ny[1:-2]

            tmpx=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outslewx,thisarmx,inslewx if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpy=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outslewy,thisarmy,inslewy if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpslew=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),np.ones(len(outslewy)),np.zeros(len(thisarmy)),np.ones(len(inslewy)) if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            

            compositex.append(tmpy if kind=='rastery' else tmpx)
            compositey.append(tmpx if kind=='rastery' else tmpy)
            compositeslew.append(tmpslew)
            flatx.extend(tmpy if kind=='rastery' else tmpx)
            flaty.extend(tmpx if kind=='rastery' else tmpy)
            flatslew.extend(tmpslew)
    elif kind=='circle':#circle pattern(s) start at origin, do circle, then back to origin or to next circle
        if tottime<0:#neg number of 8s samples around perimeter plus two
        # does extra 2 samples of pattern because timing of dumps may partially overlap slews
            n8sdumpsperrev=-tottime
        else:
            n8sdumpsperrev=None
        narms=trackinterval#should possibly be calculated from tottime, but usage of circle is very limited; envisaged for referencepointing only
        compositex=[]
        compositey=[]
        compositeslew=[]
        flatx=[]
        flaty=[]
        flatslew=[]
        for arm in range(narms):
            thisrad=radextent*(narms-arm)/(narms)
            thisperimeter=2*np.pi*thisrad#degrees
            if n8sdumpsperrev is not None:
                theta=2*np.pi*np.linspace(-0.5/n8sdumpsperrev,1+(1.5)/n8sdumpsperrev,int(np.ceil(8/sampletime*(n8sdumpsperrev+2))),endpoint=False)
            else:
                theta=np.linspace(0,2*np.pi,int(thisperimeter/(scanspeed*sampletime)))
            if (arm%2):#alternates direction with every scan
                theta=-theta
            thisarmx=thisrad*np.sin(theta)
            thisarmy=thisrad*np.cos(theta)
            if (arm%trackinterval==0):
                lastarmx=np.zeros(2)
                lastarmy=np.zeros(2)
            else:
                lastrad=radextent*(narms-arm+1)/(narms)
                lastperimeter=2*np.pi*lastrad#degrees
                if n8sdumpsperrev is not None:
                    theta=2*np.pi*np.linspace(-0.5/n8sdumpsperrev,1+(1.5)/n8sdumpsperrev,int(np.ceil(8/sampletime*(n8sdumpsperrev+2))),endpoint=False)
                else:
                    theta=np.linspace(0,2*np.pi,int(lastperimeter/(scanspeed*sampletime)))
                if (arm-1)%2:
                    theta=-theta
                lastarmx=lastrad*np.sin(theta)
                lastarmy=lastrad*np.cos(theta)
            if ((arm+1)%trackinterval==0) or arm==narms-1:
                nextarmx=np.zeros(2)
                nextarmy=np.zeros(2)
            else:
                nextrad=radextent*(narms-arm-1)/(narms)
                nextperimeter=2*np.pi*nextrad#degrees
                if n8sdumpsperrev is not None:
                    theta=2*np.pi*np.linspace(-0.5/n8sdumpsperrev,1+(1.5)/n8sdumpsperrev,int(np.ceil(8/sampletime*(n8sdumpsperrev+2))),endpoint=False)
                else:
                    theta=np.linspace(0,2*np.pi,int(nextperimeter/(scanspeed*sampletime)))
                if (arm+1)%2:
                    theta=-theta
                nextarmx=nextrad*np.sin(theta)
                nextarmy=nextrad*np.cos(theta)
            nslew=np.sqrt((lastarmx[-1]-thisarmx[0])**2+(lastarmy[-1]-thisarmy[0])**2)/(slewspeed*sampletime)
            indep=[lastarmx[-2],lastarmy[-2],lastarmx[-1],lastarmy[-1],thisarmx[0],thisarmy[0],thisarmx[1],thisarmy[1],nslew+3]
            fitter=NonLinearLeastSquaresFit(bezierpathcost,[0.,0.])
            fitter.fit(indep,np.zeros(6))
            params=fitter.params
            nx,ny=bezierpath(params,indep)
            outslewx,outslewy=nx[1:-2],ny[1:-2]

            nslew=np.sqrt((thisarmx[-1]-nextarmx[0])**2+(thisarmy[-1]-nextarmy[0])**2)/(slewspeed*sampletime)
            indep=[thisarmx[-2],thisarmy[-2],thisarmx[-1],thisarmy[-1],nextarmx[0],nextarmy[0],nextarmx[1],nextarmy[1],nslew+3]
            fitter=NonLinearLeastSquaresFit(bezierpathcost,[0.,0.])
            fitter.fit(indep,np.zeros(6))
            params=fitter.params
            nx,ny=bezierpath(params,indep)
            inslewx,inslewy=nx[1:-2],ny[1:-2]

            tmpx=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outslewx,thisarmx,inslewx if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpy=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),outslewy,thisarmy,inslewy if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]
            tmpslew=np.r_[np.zeros(int(tracktime/sampletime) if (arm%trackinterval==0) else 0),np.ones(len(outslewy)),np.zeros(len(thisarmy)),np.ones(len(inslewy)) if (((arm+1)%trackinterval==0) or arm==narms-1) else [],np.zeros(int(tracktime/sampletime) if (arm==narms-1) else 0)]

            compositex.append(tmpx)
            compositey.append(tmpy)
            compositeslew.append(tmpslew)
            flatx.extend(tmpx)
            flaty.extend(tmpy)
            flatslew.extend(tmpslew)

    return compositex,compositey,compositeslew #these coordinates are such that the upper part of pattern is sampled first; reverse order to sample bottom part first

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
        cx=[com[::-1] for com in compositex[::-1]]
        cy=[com[::-1] for com in compositey[::-1]]
    meanelev=np.zeros(len(cx))
    minsunangle=np.zeros(len(cx))
    for iarm in range(len(cx)):#spiral arm index
        scan_data,clipping_occurred = gen_scan(starttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=clip_safety_margin,min_elevation=min_elevation,max_elevation=max_elevation)
        meanelev[iarm]=np.mean(scan_data[:,2])
        #if sun elevation below 0, horizon, then rather regard sunangle as 180 degrees; note katpoint functions returns radians
        minsunangle[iarm]=np.min([target.separation(target_sun,katpoint.Timestamp(timestamp),antenna=arraycenter_antenna) if target_sun.azel(timestamp=katpoint.Timestamp(timestamp),antenna=arraycenter_antenna)[1]>0 else np.pi for timestamp in scan_data[:,0]])*180/np.pi
        starttime=scan_data[-1,0]
        if clipping_occurred:
            return False, rising, starttime-now, meanelev[iarm], minsunangle[iarm]
    return True, rising, starttime-now, np.mean(meanelev), np.mean(minsunangle)

if __name__=="__main__":
    # Set up standard script options
    parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                     description='This script performs a holography scan on the specified target. '
                                                 'All the antennas initially track the target, whereafter a subset '
                                                 'of the antennas (the "scan antennas" specified by the --scan-ants '
                                                 'option) perform a scan on the target. Note also some '
                                                 '**required** options below. Targets ordered by preference.')
    # Add experiment-specific options
    parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna). Could also be GroupA or GroupB to select half of available antennas automatically. GroupAB to alternate between GroupA and GroupB. Can specify scan antennas of particular interest to be always scanning in all cycles like: GroupAB,m013,m060. An integer specifies number of scanning antennas, chosen automatically.',default='GroupAB')
    parser.add_option('--track-ants', help='Subset of all antennas that will track source. An integer specifies number of tracking antennas, chosen automatically. (default=all non-scanning antennas)')
    parser.add_option('--num-cycles', type='int', default=-1,
                      help='Number of beam measurement cycles to complete (default=%default) use -1 for indefinite')
    parser.add_option('--cycle-duration', type='float', default=1800,
                      help='Time to spend measuring beam pattern per cycle, in seconds (default=%default)')
    parser.add_option('-l', '--scan-extent', type='float', default=10,
                      help='Diameter of beam pattern to measure, in degrees (default=%default)')
    parser.add_option('--kind', type='string', default='spiral',
                      help='Kind could be "spiral", "radial", "raster", "rastery" (default=%default)')
    parser.add_option('--tracktime', type='float', default=10,
                      help='Extra time in seconds for scanning antennas to track when passing over target (default=%default)')
    parser.add_option('--trackinterval', type='int', default=1,
                      help='track target for tracktime every this many scans (default=%default)')
    parser.add_option('--cycle-tracktime', type='float', default=30,
                      help='Extra time in seconds for scanning antennas to track when passing over target (default=%default)')
    parser.add_option('--slowtime', type='float', default=6,
                      help='Time in seconds to slow down at start and end of each spiral arm (default=%default)')
    parser.add_option('--sampletime', type='float', default=0.25,
                      help='time in seconds to spend on each sample point generated (default=%default)')
    parser.add_option('--scanspeed', type='float', default=0.1,
                      help='scan speed in degrees per second (default=%default)')
    parser.add_option('--slewspeed', type='float', default=-1,
                      help='speed at which to slew in degrees per second, or if negative number then this multiplied by scanspeed (default=%default)')
    parser.add_option('--twistfactor', type='float', default=1,
                      help='spiral twist factor (0 for straight radial, 1 standard spiral) (default=%default)')
    parser.add_option('--high-elevation-slowdown-factor', type='float', default=2.0,
                      help='factor by which to slow down nominal scanning speed at 90 degree elevation, linearly scaled from factor of 1 at 60 degrees elevation (default=%default)')
    parser.add_option('--elevation-histogram', type='string', default='',
                      help='A string of 15 comma separated count values representing a histogram in 5 degree intervals from 15 to 90 degrees elevation of known measurements (default=%default). A preferred target making the biggest impact to flatten the histogram will be selected.')
    parser.add_option('--prepopulatetime', type='float', default=10.0,
                      help='time in seconds to prepopulate buffer in advance (default=%default)')
    parser.add_option('--auto-delay', type='string', default=None,
                      help='Set CBF auto-delay on or off (default=%default)')
    parser.add_option('--dump-file', type='string', default='profile.csv',
                      help='Name of CSV file in which to dump the scan profile, when executing in test mode.')
                  
    # Set default value for any option (both standard and experiment-specific options)
    parser.set_defaults(description='Spiral holography scan', quorum=1.0, nd_params='off')
    # Parse the command line
    opts, args = parser.parse_args()

    compositex,compositey,compositeslew=generatepattern(totextent=opts.scan_extent,tottime=opts.cycle_duration,tracktime=opts.tracktime,slowtime=opts.slowtime,sampletime=opts.sampletime,scanspeed=opts.scanspeed,slewspeed=opts.slewspeed,twistfactor=opts.twistfactor,trackinterval=opts.trackinterval,kind=opts.kind)
    if testmode:
        plt.figure()
        x=[]
        y=[]
        sl=[]
        for iarm in range(len(compositex)):
            plt.plot(compositex[iarm],compositey[iarm],'.')
            x.extend(compositex[iarm])
            y.extend(compositey[iarm])
            sl.extend(compositeslew[iarm])
        x=np.array(x)
        y=np.array(y)
        sl=np.array(sl)
        for iarm in range(len(compositex)):
            slewindex=np.nonzero(compositeslew[iarm])[0]
            plt.plot(compositex[iarm][slewindex],compositey[iarm][slewindex],'.k',ms=1)
        plt.ylim([-opts.scan_extent/2,opts.scan_extent/2])
        plt.axis('equal')
        plt.title('%s scans: %d total time: %.1fs slew: %.1fs'%(opts.kind,len(compositex),len(sl)*opts.sampletime,np.sum(sl)*opts.sampletime))
        slewindex=np.nonzero(sl)[0]
        t=np.arange(len(x))*np.float64(opts.sampletime)
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t[slewindex],x[slewindex],'r.')
        plt.plot(t[slewindex],y[slewindex],'r.')
        plt.plot(t,x,'-')
        plt.plot(t,y,'--')
        plt.ylabel('[degrees]')
        plt.legend(['x','y'])
        plt.title('Position profile')
        plt.subplot(3,1,2)
        plt.plot(t[slewindex],(np.diff(x)/opts.sampletime)[slewindex],'r.')
        plt.plot(t[slewindex],(np.diff(y)/opts.sampletime)[slewindex],'r.')
        plt.plot(t[:-1],np.diff(x)/opts.sampletime,'-')
        plt.plot(t[:-1],np.diff(y)/opts.sampletime,'--')
        plt.ylabel('[degrees/s]')
        plt.legend(['dx','dy'])
        plt.title('Speed profile')
        plt.subplot(3,1,3)
        plt.plot(t[slewindex-1],(np.diff(np.diff(x))/opts.sampletime/opts.sampletime)[slewindex-1],'r.')
        plt.plot(t[slewindex-1],(np.diff(np.diff(y))/opts.sampletime/opts.sampletime)[slewindex-1],'r.')
        plt.plot(t[:-2],np.diff(np.diff(x))/opts.sampletime/opts.sampletime,'-')
        plt.plot(t[:-2],np.diff(np.diff(y))/opts.sampletime/opts.sampletime,'--')
        plt.ylabel('[degrees/s^2]')
        plt.legend(['ddx','ddy'])
        plt.title('Acceleration profile')
        plt.figure()
        scanaz_rad,scanel_rad=plane_to_sphere_holography(45.*np.pi/180.,45.*np.pi/180.,x*np.pi/180. ,y*np.pi/180.)
        scanaz=scanaz_rad*180./np.pi
        scanel=scanel_rad*180./np.pi
        plt.subplot(3,1,1)
        plt.plot(t,scanaz,'-')
        plt.plot(t,scanel,'--')
        plt.title('Position profile')
        plt.legend(['az','el'])
        plt.ylabel('[degrees]')
        plt.subplot(3,1,2)
        plt.plot(t[:-1],np.diff(scanaz)/opts.sampletime,'-')
        plt.plot(t[:-1],np.diff(scanel)/opts.sampletime,'--')
        plt.ylabel('[degrees/s]')
        plt.legend(['daz/dt','del/dt'])
        plt.title('Speed profile')
        plt.subplot(3,1,3)
        plt.plot(t[:-2],np.diff(np.diff(scanaz))/opts.sampletime/opts.sampletime,'-')
        plt.plot(t[:-2],np.diff(np.diff(scanel))/opts.sampletime/opts.sampletime,'--')
        plt.ylabel('[degrees/s^2]')
        plt.legend(['daz/dt^2','del/dt^2'])
        plt.title('Acceleration profile')
        plt.xlabel('time [s]')
        plt.show()
        import csv
        with open(opts.dump_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['time from start [sec]','azimuth [deg]','elevation [deg]'])
            for _t,_a,_e in zip(t,scanaz,scanel):
                writer.writerow(["%.6f"%_ for _ in [_t,_a,_e]])
    else:
        if len(args)==0 or args[0]=='lbandtargets':#lband targets, in order of brightness
            args=['3C 273','PKS 0408-65','PKS 1934-63','Hyd A','3C 279','PKS 0023-26','J0825-5010','PKS J1924-2914']
        elif args[0]=='sbandtargets':#sband targets, in order of brightness in S4 band
            args=['3C 454.3','PKS J1924-2914','PKS 1934-63','3C 279','PKS 2134+004','PKS 0723-008','PKS 0408-65','PKS 1421-490','PKS 0023-26','J0825-5010']
        #useful targets might not exist in catalogue
        ensure_cat={'3C 273':'J1229+0203 | *3C 273 | PKS 1226+02,radec, 12:29:06.70,  +02:03:08.6',
        'PKS 1934-63':'J1939-6342 | *PKS 1934-63,radec, 19:39:25.03,  -63:42:45.7',
        '3C 279':'J1256-0547 | *3C 279 | PKS 1253-05,radec, 12:56:11.17,  -05:47:21.5',
        'PKS 0408-65':'J0408-6545 | *PKS 0408-65,radec, 04:08:20.38,  -65:45:09.1',
        'PKS 0023-26':'J0025-2602 | *PKS 0023-26 | OB-238,radec, 00:25:49.16,  -26:02:12.6',
        'J0825-5010':'J0825-5010,radec, 08:25:26.869, -50:10:38.4877',
        'PKS J1924-2914':'J1924-2914 | *PKS J1924-2914,radec, 19:24:51.06,  -29:14:30.1',
        'Hyd A':'J0918-1205 | *Hyd A | Hydra A | 3C 218 | PKS 0915-11, radec, 09:18:05.28,  -12:05:48.9',
        '3C 454.3':'J2253+1608 | 3C 454.3 | PKS 2251+158, radec, 22:53:57.75, 16:08:53.6',
        'PKS 0723-008':'J0725-0055 | PKS 0723-008, radec, 07:25:50.64, -00:54:56.5',
        'PKS 2134+004':'J2136+0041 | PKS 2134+004, radec, 21:36:38.59, 00:41:54.2',
        'PKS 1421-490':'J1424-4913 | PKS 1421-490, radec, 14:24:32.24, -49:13:49.7'}

        # Check basic command-line options and obtain a kat object connected to the appropriate system
        with verify_and_connect(opts) as kat:
            targetnames_added=[]
            for tar in ensure_cat.keys():
                if tar not in kat.sources:
                    kat.sources.add(ensure_cat[tar])
                    targetnames_added.append(tar)
            if len(targetnames_added):
                user_logger.info("Added targets not in catalogue: %s",', '.join(targetnames_added))

            arraycenter_antenna=katpoint.Antenna('meerkat,-30:42:44.68,21:26:37.0,1038,13.5')
            catalogue = collect_targets(kat, args)
            targets=catalogue.targets
            target_sun=katpoint.Target("Sun, special")
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
                scan_ants_lower_list=opts.scan_ants.lower().replace(' ','').split(',')
                grouprange = [0]
                always_scan_ants=[]
                always_scan_ants_names=[]
                if (opts.track_ants and opts.track_ants.isdigit()):
                    GroupA,GroupB=SplitArray(np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[0] for ant in session.ants]),np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[1] for ant in session.ants]),doplot=False)
                    GroupA.extend(GroupB[::-1])
                    if int(opts.track_ants)>0:
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
                elif (scan_ants_lower_list[0]=='groupa' or scan_ants_lower_list[0]=='groupab' or scan_ants_lower_list[0]=='groupb'):
                    GroupA,GroupB=SplitArray(np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[0] for ant in session.ants]),np.array([katpoint.Antenna(ant.sensor.observer.get_value()).position_enu[1] for ant in session.ants]),doplot=False)
                    scan_ants = ant_array(kat, [session.ants[ant] for ant in (GroupA if (scan_ants_lower_list[0]=='groupa' or scan_ants_lower_list[0]=='groupab') else GroupB)], 'scan_ants')
                    if len(scan_ants_lower_list)>1:#eg. GroupAB,m000,m010
                        always_scan_ants=[ant for ant in all_ants if ant.name in scan_ants_lower_list[1:]]
                        always_scan_ants_names=[ant.name for ant in always_scan_ants]
                    if (scan_ants_lower_list[0]=='groupab'):
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
                session.obs_params['scan_ants_always']=','.join(np.sort(always_scan_ants_names))
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

                user_logger.info("Initiating %s holography scan cycles (%s %g-second "
                                 "cycles extending %g degrees) using targets %s",
                                 opts.kind,('unlimited' if opts.num_cycles<0 else '%d'%opts.num_cycles), opts.cycle_duration,
                                 opts.scan_extent, ','.join(["'%s'"%(t.name) for t in targets]))

                lasttime = time.time()
                cycle=0
                elevation_histogram=[int(val) for val in opts.elevation_histogram.split(',')] if len(opts.elevation_histogram) else []# could be length 0, default
                if len(elevation_histogram)==15:
                    user_logger.info("Using elevation_histogram: %s",opts.elevation_histogram)
                elif len(elevation_histogram)>0:
                    user_logger.warning("Supplied elevation_histogram (%s) length is %d but must be either 0 (to ignore) or 15", opts.elevation_histogram, len(elevation_histogram))
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
                        target_rising=False
                        target_elevation_cost=1e10
                        target_expected_duration=0
                        target_meanelev=0
                        target_minsunangle=0
                        target_histindex=0
                        targetinfotext=[]
                        for testtarget in targets:
                            suitable, rising, expected_duration, meanelev, minsunangle = test_target_azel_limits(testtarget,clip_safety_margin=2.0,min_elevation=opts.horizon,max_elevation=90.)
                            targetinfotext.append('%s (elev %.1f%s%s)'%(testtarget.name,meanelev,', sun %.1f'%minsunangle if (minsunangle<180) else '','' if suitable else ', unsuitable'))
                            if suitable:
                                if len(elevation_histogram)==15:#by design this histogram is meant to have 15 bins, from 15 to 90 deg elevation in 5 degree intervals
                                    histindex=int(np.clip((meanelev-15.0)/(90.-15.)*15,0,14))
                                    #ignore histogram ordering by up to 10 points maximum for sun angle ordering: 0sunangle_deg=>+10points 90sunangle_deg=>0 points
                                    suncost=10*np.clip((90-minsunangle)/90,0,1)
                                    if target_elevation_cost>elevation_histogram[histindex]+suncost:#find target with lowest histogram reading
                                        target=testtarget
                                        target_rising=rising
                                        target_expected_duration=expected_duration
                                        target_meanelev=meanelev
                                        target_minsunangle=minsunangle
                                        target_histindex=histindex
                                        target_elevation_cost=elevation_histogram[histindex]+suncost
                                else:
                                    target=testtarget
                                    target_rising=rising
                                    target_expected_duration=expected_duration
                                    target_meanelev=meanelev
                                    target_minsunangle=minsunangle
                                    break
                        user_logger.info("Targets considered: %s"%(', '.join(targetinfotext)))
                        if target is None:
                            user_logger.info("Quitting because none of the preferred targets are up")
                            break
                        else:
                            user_logger.info("Using target '%s' (mean elevation %.1f degrees)",target.name,target_meanelev)
                            user_logger.info("Current scan estimated to complete at UT %s (in %.1f minutes)",time.ctime(time.time()+target_expected_duration+time.timezone),target_expected_duration/60.)

                        target.tags = target.tags[:1] # this is to avoid overloading the cal pipeline
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
                        if (target_rising):#target is rising - scan top half of pattern first
                            cx=compositex
                            cy=compositey
                            cs=compositeslew
                        else:  #target is setting - scan bottom half of pattern first
                            cx=[com[::-1] for com in compositex[::-1]]
                            cy=[com[::-1] for com in compositey[::-1]]
                            cs=[com[::-1] for com in compositeslew[::-1]]
                        user_logger.info("Using Track antennas: %s",' '.join([ant.name for ant in track_ants if ant.name not in always_scan_ants_names]))
                        lasttime = time.time()
                        for iarm in range(len(cx)):#spiral arm index
                            user_logger.info("Performing scan arm %d of %d.", iarm + 1, len(cx))
                            user_logger.info("Using Scan antennas: %s %s",
                                             ' '.join(always_scan_ants_names),' '.join([ant.name for ant in scan_ants if ant.name not in always_scan_ants_names]))
                            for iant,scan_ant in enumerate(scan_ants):
                                session.ants = scan_ants_array[iant]
                                target.antenna = scan_observers[iant]
                                scan_data, clipping_occurred = gen_scan(lasttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                                if not kat.dry_run:
                                    if clipping_occurred:
                                        user_logger.info("Warning unexpected clipping occurred in scan pattern")
                                    session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                            for iant,track_ant in enumerate(track_ants):#also include always_scan_ants in track_ant list
                                if track_ant.name not in always_scan_ants_names:
                                    continue
                                session.ants = track_ants_array[iant]
                                target.antenna = track_observers[iant]
                                scan_data, clipping_occurred = gen_scan(lasttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                                if not kat.dry_run:
                                    if clipping_occurred:
                                        user_logger.info("Warning unexpected clipping occurred in scan pattern")
                                    session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                            
                            lastisslew=None#so that first sample's state is also recorded
                            for it in range(len(cx[iarm])):
                                if cs[iarm][it]!=lastisslew:
                                    lastisslew=cs[iarm][it]
                                    session.telstate.add('obs_label','slew' if lastisslew else '%d.%d.%d'%(cycle,igroup,iarm),ts=scan_data[it,0])
                                
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
                        if len(elevation_histogram)==15:#by design this histogram is meant to have 15 bins, from 15 to 90 deg elevation in 5 degree intervals
                            elevation_histogram[target_histindex]+=1#update histogram as we go along
                        user_logger.info("Safe to interrupt script now if necessary")
                        if kat.dry_run:#only test one group - dryrun takes too long and causes CAM to bomb out
                            user_logger.info("Testing only one group for dry-run")
                            break
                    if kat.dry_run:#only test one cycle - dryrun takes too long and causes CAM to bomb out
                        user_logger.info("Testing only cycle for dry-run")
                        break
                    cycle+=1
