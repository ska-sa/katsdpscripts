#!/usr/bin/python
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

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
import katpoint

# Import script helper functions from observe.py
from katcorelib import standard_script_options, verify_and_connect, collect_targets, \
                       start_session, user_logger, ant_array
import numpy as np
import scipy
from scikits.fitting import NonLinearLeastSquaresFit, PiecewisePolynomial1DFit

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
#slowtime redistributes samples on each arm so that start and stop of scan occurs slower within this timerange in seconds
def generatespiral(totextent,tottime,tracktime=1,slewtime=1,slowtime=1,sampletime=1,kind='uniform',mirrorx=False):
    totextent=np.float(totextent)
    tottime=np.float(tottime)
    sampletime=np.float(sampletime)
    nextrazeros=int(np.float(tracktime)/sampletime)
    nextraslew=int(np.float(slewtime)/sampletime)
    print 'nextrazeros',nextrazeros+nextraslew
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
    else:#'uniform'
        c=180.0/(16.0*np.pi)
        narms=2*int(np.sqrt(tottime/c+((tracktime+slewtime)/c)**2)-(tracktime+slewtime)/c)#ensures even number of arms - then scan pattern ends on target (if odd it will not)
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
            interx=np.linspace(compositex[ia][-1],compositex[ia+1][0],2+nextraslew*2)
            compositex[ia]=np.r_[compositex[ia],interx[1:1+nextraslew]]
            compositex[ia+1]=np.r_[interx[1+nextraslew:-1],compositex[ia+1]]
            intery=np.linspace(compositey[ia][-1],compositey[ia+1][0],2+nextraslew*2)
            compositey[ia]=np.r_[compositey[ia],intery[1:1+nextraslew]]
            compositey[ia+1]=np.r_[intery[1+nextraslew:-1],compositey[ia+1]]
            ninterx=np.linspace(ncompositex[ia][-1],ncompositex[ia+1][0],2+nextraslew*2)
            ncompositex[ia]=np.r_[ncompositex[ia],ninterx[1:1+nextraslew]]
            ncompositex[ia+1]=np.r_[ninterx[1+nextraslew:-1],ncompositex[ia+1]]
            nintery=np.linspace(ncompositey[ia][-1],ncompositey[ia+1][0],2+nextraslew*2)
            ncompositey[ia]=np.r_[ncompositey[ia],nintery[1:1+nextraslew]]
            ncompositey[ia+1]=np.r_[nintery[1+nextraslew:-1],ncompositey[ia+1]]
    if (mirrorx):
        for ia in range(narms):
            compositex[ia]=-compositex[ia]
            ncompositex[ia]=-ncompositex[ia]
    
    return compositex,compositey,ncompositex,ncompositey

def gen_scan(lasttime,target,az_arm,el_arm,timeperstep):
    num_points = np.shape(az_arm)[0]
    az_arm = az_arm*np.pi/180.0
    el_arm = el_arm*np.pi/180.0
    scan_data = np.zeros((num_points,3))    
    attime = lasttime+np.arange(1,num_points+1)*timeperstep
    #spiral arm scan
    targetaz_rad,targetel_rad=target.azel(attime)
    scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,az_arm ,el_arm )
    scan_data[:,0] = attime
    scan_data[:,1] = katpoint.wrap_angle(scanaz)*180.0/np.pi
    scan_data[:,2] = scanel*180.0/np.pi
    return scan_data

def gen_track(attime,target):
    track_data = np.zeros((len(attime),3))
    targetaz_rad,targetel_rad=target.azel(attime)
    track_data[:,0] = attime
    track_data[:,1] = katpoint.wrap_angle(targetaz_rad)*180.0/np.pi
    track_data[:,2] = targetel_rad*180.0/np.pi
    return track_data


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='This script performs a holography scan on the specified target. '
                                             'All the antennas initially track the target, whereafter a subset '
                                             'of the antennas (the "scan antennas" specified by the --scan-ants '
                                             'option) perform a spiral raster scan on the target. Note also some '
                                             '**required** options below.')
# Add experiment-specific options
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
parser.add_option('--slewtime', type='float', default=1.0,
                  help='Extra time in seconds for scanning antennas to slew when passing from one spiral arm to another (default=%default)')
parser.add_option('--slowtime', type='float', default=1.0,
                  help='Time in seconds to slow down at start and end of each spiral arm (default=%default)')
parser.add_option('--sampletime', type='float', default=1.0,
                  help='time in seconds to spend on pointing (default=%default)')
parser.add_option('--prepopulatetime', type='float', default=30.0,
                  help='time in seconds to prepopulate buffer in advance (default=%default)')
parser.add_option('--mirrorx', action="store_true", default=False,
                  help='Mirrors x coordinates of pattern (default=%default)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Spiral holography scan', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

compositex,compositey,ncompositex,ncompositey=generatespiral(totextent=opts.scan_extent,tottime=opts.cycle_duration,tracktime=opts.tracktime,slewtime=opts.slewtime,slowtime=opts.slowtime,sampletime=opts.sampletime,kind=opts.kind,mirrorx=opts.mirrorx)

if len(args) == 0:
    raise ValueError("Please specify a target argument via name ('Ori A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check basic command-line options and obtain a kat object connected to the appropriate system
with verify_and_connect(opts) as kat:
    catalogue = collect_targets(kat, args)
    targets=catalogue.targets
    if len(targets) == 0:
        raise ValueError("Please specify a target argument via name ('Ori A'), "
                         "description ('azel, 20, 30') or catalogue file name ('sources.csv')")
    target=targets[0]#only use first target
    lasttargetel=target.azel()[1]*180.0/np.pi
    # Initialise a capturing session (which typically opens an HDF5 file)
    with start_session(kat, **vars(opts)) as session:
        # Use the command-line options to set up the system
        session.standard_setup(**vars(opts))
        if not opts.no_delays and not kat.dry_run :
            if session.dbe.req.auto_delay('on'):
                user_logger.info("Turning on delay tracking.")
            else:
                user_logger.error('Unable to turn on delay tracking.')
        elif opts.no_delays and not kat.dry_run:
            if session.dbe.req.auto_delay('off'):
                user_logger.info("Turning off delay tracking.")
            else:
                user_logger.error('Unable to turn off delay tracking.')
            if session.dbe.req.zero_delay():
                user_logger.info("Zeroed the delay values.")
            else:
                user_logger.error('Unable to zero delay values.')

        all_ants = session.ants
        # Form scanning antenna subarray (or pick the first antenna as the default scanning antenna)
        scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else session.ants[0], 'scan_ants')
        # Assign rest of antennas to tracking antenna subarray
        track_ants = ant_array(kat, [ant for ant in all_ants if ant not in scan_ants], 'track_ants')
        # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
        nd_params = session.nd_params
        session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
        session.capture_start()
        session.label('holo')
        user_logger.info("Initiating spiral holography scan cycles (%d %g-second cycles extending %g degrees) on target '%s'"
                         % (opts.num_cycles, opts.cycle_duration, opts.scan_extent, target.name))
        session.set_target(target)
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
            #print("Using all antennas: %s" % (' '.join([ant  for ant in ants]),))
            user_logger.info("Using all antennas: %s" % (' '.join([ant.name  for ant in session.ants]),))
            slewtime = 0
            session.ants = all_ants
            #get both antennas to target ASAP
            scan_track = gen_track(np.arange(opts.prepopulatetime)+time.time(),target)
            session.load_scan(scan_track[:,0],scan_track[:,1],scan_track[:,2])
            lasttime=time.time()+opts.prepopulatetime
            for iarm in range(len(cx)):#spiral arm index
                user_logger.info("Performing scan arm %d of %d."%(iarm+1,len(cx)))
                scan_data = gen_scan(lasttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime)
                scan_track = gen_track(scan_data[:,0],target)
                session.ants = scan_ants
                user_logger.info("Using Scan antennas: %s" % (' '.join([ant.name  for ant in session.ants]),))
                session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                session.ants = track_ants
                user_logger.info("Using Track antennas: %s" % (' '.join([ant.name  for ant in session.ants]),))
                session.load_scan(scan_track[:,0],scan_track[:,1],scan_track[:,2])
                time.sleep(scan_data[-1,0]-time.time()-opts.prepopulatetime)
                lasttime = scan_data[-1,0]
        time.sleep(lasttime-time.time()+1.0)#wait for 1 second more than timestamp for last coordinate
        #set session antennas to all so that stow-when-done option will stow all used antennas and not just the scanning antennas
        session.ants = all_ants

