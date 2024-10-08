#!/usr/bin/python
#track target(s) for a specified time.

from katcorelib import standard_script_options, verify_and_connect,  user_logger , ant_array
import katpoint
import time

import numpy as np
#import scipy
from scikits.fitting import NonLinearLeastSquaresFit

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

#note that we want spiral to only extend to above horizon for first few scans in case source is rising
#should test if source is rising or setting before each composite scan, and use -compositey if setting
def generatespiral(totextent,tottime,tracktime=1,sampletime=1,kind='uniform',mirrorx=False):
    totextent=np.float(totextent)
    tottime=np.float(tottime)
    sampletime=np.float(sampletime)
    nextrazeros=int(np.float(tracktime)/sampletime)
    print 'nextrazeros',nextrazeros
    tracktime=nextrazeros*sampletime
    radextent=np.float(totextent)/2.0
    if (kind=='dense-core'):
        c=np.sqrt(2)*180.0/(16.0*np.pi)
        narms=2*int(np.sqrt(tottime/c+(tracktime/c)**2)-tracktime/c)#ensures even number of arms - then scan pattern ends on target (if odd it will not)
        ntime=int((tottime-tracktime*narms)/(sampletime*narms))
        armrad=radextent*(np.linspace(0,1,ntime))
        armtheta=np.linspace(0,np.pi,ntime)
        armx=armrad*np.cos(armtheta)
        army=armrad*np.sin(armtheta)
    elif (kind=='approx'):
        c=180.0/(16.0*np.pi)
        narms=2*int(np.sqrt(tottime/c+(tracktime/c)**2)-tracktime/c)#ensures even number of arms - then scan pattern ends on target (if odd it will not)
        ntime=int((tottime-tracktime*narms)/(sampletime*narms))
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
    else:#'uniform'
        c=180.0/(16.0*np.pi)
        narms=2*int(np.sqrt(tottime/c+(tracktime/c)**2)-tracktime/c)#ensures even number of arms - then scan pattern ends on target (if odd it will not)
        ntime=int((tottime-tracktime*narms)/(sampletime*narms))
        armx=np.zeros(ntime)
        army=np.zeros(ntime)
        #must be on curve x=t*cos(np.pi*t),y=t*sin(np.pi*t)
        #intersect (x-x0)**2+(y-y0)**2=1/ntime**2 with spiral
        lastr=0.0
        for it in range(1,ntime):
            data=np.array([1.0/ntime])
            indep=np.array([armx[it-1],army[it-1]])#last calculated coordinate in arm, is x0,y0
            initialparams=np.array([lastr+1.0/ntime]);
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
        if (mirrorx):
            compositex[ia]=-x
            compositey[ia]=y
            ncompositex[ia]=-nx
            ncompositey[ia]=ny
        else:
            compositex[ia]=x
            compositey[ia]=y
            ncompositex[ia]=nx
            ncompositey[ia]=ny

    return compositex,compositey,ncompositex,ncompositey



def track(ants, target, duration=10, dry_run=False):
    print "Target  :",target
    # send this target to the antenna.
    ants.req.target(target.description)
    print "Target desc. : ",target.description
    ant_names = ','.join([ant.name for ant in ants])
    ants.req.mode("POINT")
    user_logger.info("Slewing %s to target : %r" % (ant_names, target))
    #Wait for antenna to lock onto target
    locks=0
    if not dry_run :
    	for ant_x in ants:
            if ant_x.wait('lock', True, timeout=300): locks +=1
            print "locks status:", ant_x.name , locks,len(ants)
    else:
        locks = len(ants)
    if len(ants)==locks:
        user_logger.info("Target reached : %r wait for %d seconds before slewing to the next target" % (target, duration))
        if not dry_run : time.sleep(duration)
        user_logger.info("Test complete : %r" % (target,))
        return True
    else:
        user_logger.warning("Unable to track Target : %r Check %s sensors  " % (target,ant_names))
        return False


# Set up standard script options
parser = standard_script_options(usage="%prog [options] 'target'",
                                 description='This script performs a holography scan on the specified target. '
                                             'All the antennas initially track the target, whereafter a subset '
                                             'of the antennas (the "scan antennas" specified by the --scan-ants '
                                             'option) perform a spiral raster scan on the target. Note also some '
                                             '**required** options below.')
parser.add_option('-b', '--scan-ants', help='Subset of all antennas that will do raster scan (default=first antenna)')
parser.add_option('--num-cycles', type='int', default=1,
                  help='Number of beam measurement cycles to complete (default=%default)')
parser.add_option('--cycle-duration', type='float', default=300.0,
                  help='Time to spend measuring beam pattern per cycle, in seconds (default=%default)')
parser.add_option('-l', '--scan-extent', type='float', default=4.0,
                  help='Diameter of beam pattern to measure, in degrees (default=%default)')
parser.add_option('--kind', type='string', default='uniform',
                  help='Kind of spiral, could be "uniform" or "dense-core" (default=%default)')
parser.add_option('--tracktime', type='float', default=1.0,
                  help='Extra time in seconds for scanning antennas to track when passing over target (default=%default)')
parser.add_option('--sampletime', type='float', default=1.0,
                  help='time in seconds to spend on pointing (default=%default)')
parser.add_option('--mirrorx', action="store_true", default=False,
                  help='Mirrors x coordinates of pattern (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Spiral holography scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()
if opts.observer is None :  raise RuntimeError("No observer provided script")

if len(args) == 0:
    raise ValueError("Please specify a target argument via name ('Ori A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")
else:
    target = katpoint.Target(args[0])

compositex,compositey,ncompositex,ncompositey=generatespiral(totextent=opts.scan_extent,tottime=opts.cycle_duration,tracktime=opts.tracktime,sampletime=opts.sampletime,kind=opts.kind,mirrorx=opts.mirrorx)
timeperstep=opts.sampletime;


targetlist = []
for argstr in args:
    temp,taz,tel= argstr.split(',')
   
    print "azimuth : %r  Elevation : %r " % (taz, tel)
    targetlist.append([taz,tel])

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
   # observation_sources = collect_targets(kat, args)
    print "Set Sensor stratergy"
    lasttargetel=target.azel()[1]*180.0/np.pi
    kat.ants.set_sampling_strategy("lock", "event")
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        if not kat.dry_run :
            user_logger.error("Unable to set Antenna mode to 'STOP'.")

    all_ants = kat.ants
    # Form scanning antenna subarray (or pick the first antenna as the default scanning antenna)
    scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else kat.ants[0], 'scan_ants')
    # Assign rest of antennas to tracking antenna subarray
    track_ants = ant_array(kat, [ant for ant in all_ants if ant not in scan_ants], 'track_ants')
    # Disable noise diode by default (to prevent it firing on scan antennas only during scans)

    user_logger.info("Initiating spiral holography scan cycles (%d %g-second cycles extending %g degrees) on target '%s'"
                     % (opts.num_cycles, opts.cycle_duration, opts.scan_extent, target.name))

    for cycle in range(opts.num_cycles):
        targetel=target.azel()[1]*180.0/np.pi
        if (targetel>lasttargetel):#target is rising - scan top half of pattern first
            cx=compositex
            cy=compositey
            if (targetel<opts.horizon):
                user_logger.info("Exiting because target is %g degrees below horizon limit of %g."%((opts.horizon-targetel),opts.horizon))
                break;# else it is ok that target just above horizon limit
        else:#target is setting - scan bottom half of pattern first
            cx=ncompositex
            cy=ncompositey
            if (targetel<opts.horizon+(opts.scan_extent/2.0)):
                user_logger.info("Exiting because target is %g degrees too low to accommodate a scan extent of %g degrees above the horizon limit of %g."%((opts.horizon+(opts.scan_extent/2.0)-targetel),opts.scan_extent,opts.horizon))
                break;
        user_logger.info("Performing scan cycle %d."%(cycle+1))
        lasttargetel=targetel
        
        user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in all_ants]),))
        track(all_ants,target,duration=0,dry_run=kat.dry_run)
        #session.fire_noise_diode(announce=False, **nd_params)#provides opportunity to fire noise diode
        
        user_logger.info("Using scan antennas: %s" % (' '.join([ant.name for ant in scan_ants]),))
#                session.set_target(target)
#                session.ants.req.drive_strategy('shortest-slew')
#                session.ants.req.mode('POINT')
        for iarm in range(len(cx)):#spiral arm index
            scan_index=0
            wasstowed=False
            while(scan_index!=len(cx[iarm])-1):
                while (not kat.dry_run and wasstowed):
                    user_logger.info("Attempting to recover from wind stow" )
                    user_logger.info("Using all antennas: %s" % (' '.join([ant.name for ant in all_ants]),))
                    track(all_ants,target,duration=0,dry_run=kat.dry_run)
                    if (not any([res._returns[0][4]=='STOW' for res in all_ants.req.sensor_value('mode').values()])):
                        scan_index=0
                        wasstowed=False
                        #session.fire_noise_diode(announce=False, **nd_params)#provides opportunity to fire noise diode
                        user_logger.info("Using scan antennas: %s" % (' '.join([ant.name for ant in scan_ants]),))
                        if (cx[iarm][scan_index]!=0.0 or cy[iarm][scan_index]!=0.0):
                            targetaz_rad,targetel_rad=target.azel()
                            scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,cx[iarm][scan_index]*np.pi/180.0,cy[iarm][scan_index]*np.pi/180.0)
                            # targetx,targety=katpoint.sphere_to_plane[opts.projection](targetaz_rad,targetel_rad,scanaz,scanel)
                            targetx,targety=sphere_to_plane_holography(scanaz,scanel,targetaz_rad,targetel_rad)
                            scan_ants.req.offset_fixed(targetx*180.0/np.pi,-targety*180.0/np.pi,opts.projection)
                            # session.ants.req.offset_fixed(cx[iarm][scan_index],cy[iarm][scan_index],opts.projection)
                            time.sleep(10)#gives 10 seconds to slew to outside arm if that is where pattern commences
                        user_logger.info("Recovered from wind stow, repeating cycle %d scan %d"%(cycle+1,iarm+1))
                    else:
                        time.sleep(60)
                lastproctime=time.time()
                for scan_index in range(len(cx[iarm])):#spiral arm scan
                    targetaz_rad,targetel_rad=target.azel()
                    scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,cx[iarm][scan_index]*np.pi/180.0,cy[iarm][scan_index]*np.pi/180.0)
                    # targetx,targety=katpoint.sphere_to_plane[opts.projection](targetaz_rad,targetel_rad,scanaz,scanel)
                    targetx,targety=sphere_to_plane_holography(scanaz,scanel,targetaz_rad,targetel_rad)
                    scan_ants.req.offset_fixed(targetx*180.0/np.pi,-targety*180.0/np.pi,opts.projection)
                    # session.ants.req.offset_fixed(cx[iarm][scan_index],cy[iarm][scan_index],opts.projection)
                    curproctime=time.time()
                    proctime=curproctime-lastproctime
                    if (timeperstep>proctime):
                        time.sleep(timeperstep-proctime)
                    lastproctime=time.time()
                    if not kat.dry_run and (np.any([res._returns[0][4]=='STOW' for res in all_ants.req.sensor_value('mode').values()])):
                        if (wasstowed==False):
                            user_logger.info("Cycle %d scan %d interrupted. Some antennas are stowed ... waiting to resume scanning"%(cycle+1,iarm+1) )
                        wasstowed=True
                        time.sleep(60)
                        break#repeats this spiral arm scan if stow occurred
    #set session antennas to all so that stow-when-done option will stow all used antennas and not just the scanning antennas
