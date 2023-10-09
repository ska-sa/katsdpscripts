#!/usr/bin/python
# mattieu@sarao.ac.za 19 September 2023
# Script that uses katsdpcal's delay calibration visibilties and calibration solutions to measure the HV phase.
# This method is an implementation of a description of how Kim McAlpine determined the original L-band BSpline
# that is routinely used in the CAL pipeline. The method has been extended to work on unpolarised targets, and
# in other bands.

import optparse
import katpoint
import katdal
import pysolr
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, splev

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
try:
    from katholog.utilities import plot_hv_phase_spline
except:
    plot_hv_phase_spline=None
    pass

fid2fn = lambda fid: "http://archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)

def get_info(f):
    target_sun=katpoint.Target("Sun, special")
    mean_timestamp=np.mean(f.timestamps)
    info={'cbid':f.source.capture_block_id,'band':f.spectral_windows[f.spw].band,'targetname':f.catalogue.targets[0].aliases[0],
            'mean_timestamp':np.mean(f.timestamps),'nchannels':len(f.channels),'ndumps':len(f.timestamps),'dump_period':f.dump_period,
            'elevation':np.nanmean(f.el),'temperature':np.nanmean(f.temperature),'wind_speed':np.nanmean(f.wind_speed),
            'parang':np.nanmean(f.parangle),
            'sun_el':target_sun.azel(timestamp=mean_timestamp,antenna=f.ants[0])[1]*180./np.pi,
            'sun_angle':f.catalogue.targets[0].separation(target_sun,katpoint.Timestamp(mean_timestamp))*180./np.pi,
            'ref_ant':f.source.metadata.attrs['cal_refant'],'antennas':[a.name for a in f.ants],'receivers':[f.receivers[a.name] for a in f.ants]}
    return info
    
def print_info(cbid):
    fid2fn = lambda fid: "http://archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)
    f=katdal.open(fid2fn(cbid))
    info=get_info(f)
    print(info)    

#the channelisation might be 1K,4K,32K 
#xyphase.shape may be [nant,nchans] or just [nchans]
def ensure_1k(freqMHz,xyphase):
    if len(freqMHz)==1024:
        return freqMHz,xyphase
    oversample=len(freqMHz)//1024#e.g. 4 or 32
    if oversample!=len(freqMHz)/1024 or not (oversample==4 or oversample==32):
        raise RuntimeError('Unexpected number of channels: %d'%len(freqMHz))
    if len(xyphase.shape)==2:
        return freqMHz[::oversample],np.nanmedian(xyphase.reshape([xyphase.shape[0],1024,oversample]),axis=2)
    return freqMHz[::oversample],np.nanmedian(xyphase.reshape([1024,oversample]),axis=1)
    

    
#only accepts the finite complex visibilities
#optimises center, orientation and stretch but not shape!
def ellipsecost(params,vis):
    cx,cy,rot,ell=params
    x=vis.real
    y=vis.imag
    sinrot=np.sin(rot)
    cosrot=np.cos(rot)
    nx=(cosrot*(x-cx)-sinrot*(y-cy))/ell
    ny=(sinrot*(x-cx)+cosrot*(y-cy))
    # pang=np.linspace(0,np.pi,32,endpoint=False)
    pang=np.linspace(0,np.pi,8,endpoint=False)
    projectedx=np.cos(pang[np.newaxis,:])*nx[:,np.newaxis]-np.sin(pang[np.newaxis,:])*ny[:,np.newaxis]
    sortedprojectedx=np.sort(projectedx,axis=0)
    #after offset, rotation, scaling, the distribution should be circularly symmetric around median point, ignoring 20% outliers
    #now ideally these are all almost equal along axis 0
    # plot(sortedprojectedx)
    lenx=len(x)
    mid=lenx//2
    stacked=np.c_[sortedprojectedx[lenx-mid:lenx-lenx//10,:],-sortedprojectedx[mid-1:lenx//10-1:-1,:]]#eliminates median point if odd number of points
    varstacked=np.var(stacked,axis=1)
    if ell<1:
        return np.sum(varstacked)*np.exp(1-ell)
    return np.sum(varstacked)
    # plot(stdstacked)
    # print(np.sum(stdstacked[:-len(x)//10]**2))
    
def test_distribution_fit():
    from scipy.optimize import minimize

    cx=np.tile(np.nan,1024)
    cy=np.tile(np.nan,1024)
    xyphase=np.tile(np.nan,1024)
    ell=np.tile(np.nan,1024)
    for i in range(1024):
        vis=crosscalvis[:,:,:,i].reshape(-1)+0
        vis=vis[np.nonzero(np.isfinite(vis))]
        if len(vis)>20:
            x0=np.r_[median(vis.real),median(vis.imag),0,1]
            x=minimize(ellipsecost,x0,args=tuple([vis]),method='Nelder-Mead')
            cx[i],cy[i],xyphase[i],ell[i]=x.x[0],x.x[1],x.x[2]*180/pi,x.x[3]
            print(i,xyphase[i])

#filename='https://archive-gw-1.kat.ac.za/1693662868/1693662868_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNjk1NzQ1NDY5LCJwcmVmaXgiOlsiMTY5MzY2Mjg2OCJdLCJleHAiOjE2OTYzNTAyNjksInN1YiI6Im1hdHRpZXVAc2FyYW8uYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.sKchekaim32JyEYO5nu4KQ0gOHxO-UNLH_C1Aya9j5_FSwIc_VTRSgTTvjkyMQbtSYozMzPlqilJAGFD-e2b7g'
# filename='1693662868_sdp_l0.full.rdb'#lband 1k 8s 3C286
# filename='1695817735_sdp_l0.full.rdb'#lband 4k 8s 3C286
# filename='1693661748_sdp_l0.full.rdb'#uhf 1k 8s 3C286
# filename='./s0/1693657602_sdp_l0.full.rdb'#s0 1k 8s 3C286
# filename='./s0/1696840275_sdp_l0.full.rdb'#s0 1k 8s 3C286
# filename='./s2/1693660786_sdp_l0.full.rdb'#s2 1k 8s 3C286
# filename='./s4/1693658891_sdp_l0.full.rdb'#s4 1k 8s 3C286
# filename=1693496913 #s4 4k 8s 3C286
# filename='1694437059_sdp_l0.full.rdb'#l-band 1k 8s pks1934
#do_median_time=False#take median over time of uncorrected visibilities before processing
#do_compute_parallelhands=True #for displaying parallel hand histograms (slower)
def analyse_hv_phase(filename,do_median_time=False,do_compute_parallelhands=True,do_plot=True,basename=None):
    # do_median_time=False
    # do_compute_parallelhands=True
    # do_plot=True
    # basename=None
    f=katdal.open(filename)
    band=f.spectral_windows[f.spw].band
    if basename is None:
        basename=f.source.capture_block_id
    if do_compute_parallelhands:
        #select un_corrected track data, but not slew, not stop (antenna is not tracking anymore while cal is doing calculations)
        f.select(corrprods='cross', compscans='un_corrected',scans='track')#scans='track' is equiv to dumps=(f.sensor['obs_activity']=='track')
    else:
        f.select(corrprods='cross', pol='hv,vh',compscans='un_corrected',scans='track')
    corr_prod_list=f.corr_products.tolist()
    cal_pol_ordering=f.source.metadata.attrs['cal_pol_ordering']#['v', 'h']
    freqMHz=f.freqs/1e6
    #get G solution index for un_corrected visibilities track
    if (f.sensor.get('cal_product_G')[0].shape is () and #f.sensor.get('cal_product_G')[0] typically = (nan+nanj)
        f.sensor.get('cal_product_G')[f.sensor.get('cal_product_G').events[1]].shape is not ()):
        #an invalid G solution is reported at the start of the observation
        #the first valid one is applicable to the first track of un_corrected data
        idump_G=f.sensor.get('cal_product_G').events[1]
        print('Detected idump_G=%d'%idump_G)
    else:
        raise Exception('Unexpected: cal_product_G event 0 is expected to be nan, event 1 is expected to be valid')
    #get appropriate B solution index - in this case it is first event, which also actually equals dump 0
    idump_B=f.sensor.get('cal_product_B0').events[0]
    idump_K=f.sensor.get('cal_product_K').events[0]
    idump_KX=f.sensor.get('cal_product_KCROSS_DIODE').events[0]
    idump_BX=f.sensor.get('cal_product_BCROSS_DIODE0').events[0]

    ndumps=len(f.timestamps)
    print('Loading %d dumps once off...'%ndumps)
    uncorrectedvis=f.vis[:]+0#load first to be faster
    uncorrectedvis[np.nonzero(f.flags)]=np.nan
    #nanmedian over time axis upfront - even uncalibrated these remain very constant
    if do_median_time:
        uncorrectedvis=(np.nanmedian(np.real(uncorrectedvis),axis=0)+1j*np.nanmedian(np.imag(uncorrectedvis),axis=0))[np.newaxis,:,:]
        ndumps=1
    #represent in matrix compactly
    if do_compute_parallelhands:
        ucocalvis=np.tile(np.nan+1j*np.nan,[ndumps,len(f.ants),len(f.ants),len(freqMHz)])
        cocalvis=np.tile(np.nan+1j*np.nan,[ndumps,len(f.ants),len(f.ants),len(freqMHz)])
        ucrosscalvis=np.tile(np.nan+1j*np.nan,[ndumps,len(f.ants),len(f.ants),len(freqMHz)])
    crosscalvis=np.tile(np.nan+1j*np.nan,[ndumps,len(f.ants),len(f.ants),len(freqMHz)])
    #cocalvis
    #m000h-m000h, m000h-m001h, m000h-m002h,...,  m000h-m063h
    #m001v-m000v, m001h-m001h, m001h-m002h,...,  m001h-m063h
    #m002v-m000v, m002v-m001v, m002h-m002h,...,  m002h-m063h
    #m003v-m000v, m003v-m001v, m003v-m002v,...,  m003h-m063h
    #crosscalvis
    #m000h-m000v, m000h-m001v, m000h-m002v,...,  m000h-m063v
    #m001h-m000v, m001h-m001v, m001h-m002v,...,  m001h-m063v
    #m002h-m000v, m002h-m001v, m002h-m002v,...,  m002h-m063v
    #m003h-m000v, m003h-m001v, m003h-m002v,...,  m003h-m063v
    for iant in range(len(f.ants)):
        for jant in range(iant+1,len(f.ants)):
            for iorder,polorder in enumerate([[1,1],[1,0],[0,1],[0,0]]):#hh, hv, vh, vv
                if not do_compute_parallelhands:
                    if polorder[0]==polorder[1]:
                        continue                    
                ipol,jpol=polorder#0 is v, 1 is h
                product=[f.ants[iant].name+cal_pol_ordering[ipol],f.ants[jant].name+cal_pol_ordering[jpol]]
                K_delay_ns=(f.sensor.get('cal_product_K')[idump_K][ipol,iant]-f.sensor.get('cal_product_K')[idump_K][jpol,jant])*1e9
                KCROSS_DIODE_delay_ns=(np.nanmedian(f.sensor.get('cal_product_KCROSS_DIODE')[idump_KX][ipol,:],axis=-1)-np.nanmedian(f.sensor.get('cal_product_KCROSS_DIODE')[idump_KX][jpol,:],axis=-1))*1e9
                K_phase_rad=2*np.pi*freqMHz*1e6*K_delay_ns/1e9
                K=np.exp(1j*K_phase_rad)
                KCROSS_DIODE_phase_rad=2*np.pi*freqMHz*1e6*KCROSS_DIODE_delay_ns/1e9
                KCROSS_DIODE=np.exp(1j*KCROSS_DIODE_phase_rad)
                B=(f.sensor.get('cal_product_B0')[idump_B][:,ipol,iant])*np.conj(f.sensor.get('cal_product_B0')[idump_B][:,jpol,jant])
                G=f.sensor.get('cal_product_G')[idump_G][ipol,iant]*np.conj(f.sensor.get('cal_product_G')[idump_G][jpol,jant])
                BCROSS_DIODE=(np.nanmedian(f.sensor.get('cal_product_BCROSS_DIODE0')[idump_BX][:,ipol,:],axis=-1))*np.conj(np.nanmedian(f.sensor.get('cal_product_BCROSS_DIODE0')[idump_BX][:,jpol,:],axis=-1))
                productindex=corr_prod_list.index(product)
                correctedvis=uncorrectedvis[:,:,productindex]/(G*K*B*KCROSS_DIODE*BCROSS_DIODE)[np.newaxis,:]
                if iorder==0:
                    cocalvis[:,iant,jant,:]=correctedvis
                    if do_compute_parallelhands:
                        ucocalvis[:,iant,jant,:]=uncorrectedvis[:,:,productindex]
                elif iorder==1:#conjugates the HV visibilities instead of the VH ones
                    crosscalvis[:,iant,jant,:]=np.conj(correctedvis)
                    if do_compute_parallelhands:
                        ucrosscalvis[:,iant,jant,:]=np.conj(uncorrectedvis[:,:,productindex])
                elif iorder==2:
                    crosscalvis[:,jant,iant,:]=correctedvis
                    if do_compute_parallelhands:
                        ucrosscalvis[:,jant,iant,:]=uncorrectedvis[:,:,productindex]
                else:
                    cocalvis[:,jant,iant,:]=np.conj(correctedvis)
                    if do_compute_parallelhands:
                        ucocalvis[:,jant,iant,:]=np.conj(uncorrectedvis[:,:,productindex])
    #determine average HVphase estimate as center starting point for wrapping
    #only works properly if target is polarised
    if False:#f.catalogue.targets[0].aliases[0] =='3C286': 
        hist_counts=[]
        for i in range(8):
            ch=slice(i*len(f.channels)//8,(i+1)*len(f.channels)//8)
            vals=np.angle(crosscalvis[:,:,:,ch].reshape(-1))*180/pi
            valid=np.nonzero(isfinite(vals))[0]
            hist_count,bin_edge=histogram(vals[valid],bins=100,density=True)
            hist_counts.append(hist_count)
        ipeak_chanrange=np.argmax(np.max(hist_counts,axis=1))
        ipeak_hvphase_offset=np.argmax(np.max(hist_counts,axis=0))
        HV_centering_phase=bin_edge[ipeak_hvphase_offset]
        if HV_centering_phase>90:#in S-band, this might be near -180 or 180; lets force to -180
            HV_centering_phase-=360# for consistency across datasets and subbands
    else:#unpolarised target - cannot distinguish 180 degree ambiguity, use prior knowledge starting poing
        if band[0]=='U':#UHF
            HV_centering_phase=-14
        elif band[0]=='L':#Lband
            HV_centering_phase=-23.5
        else:#S-band
            HV_centering_phase=-180
            
    #classical kim solution, but using appropriate scalar branch cut
    complex_median=np.nanmedian(crosscalvis.real,axis=(0,1,2))+1j*np.nanmedian(crosscalvis.imag,axis=(0,1,2))
    phasemedian=(np.angle(complex_median)*180/pi-HV_centering_phase+180)%360-180+HV_centering_phase
    medianphase=np.nanmedian((np.angle(crosscalvis)*180/pi-HV_centering_phase+180)%360-180+HV_centering_phase,axis=(0,1,2))    

    anglecalvis=np.angle(crosscalvis[:,:,:,:])*180/pi
    if True:
        nants=anglecalvis.shape[1]
        nchans=len(freqMHz)
        HV_phase_per_ant_chan=np.tile(HV_centering_phase,[nants,nchans])
        VH_phase_per_ant_chan=np.tile(HV_centering_phase,[nants,nchans])
        convergence=[HV_centering_phase]
        for it in range(20):
            HV_phase_per_ant_chan=np.nanmedian((anglecalvis[:,:,:,:]-HV_phase_per_ant_chan[np.newaxis,:,np.newaxis,:]+90)%180-90+HV_phase_per_ant_chan[np.newaxis,:,np.newaxis,:],axis=(0,1))
            VH_phase_per_ant_chan=np.nanmedian((anglecalvis[:,:,:,:]-VH_phase_per_ant_chan[np.newaxis,np.newaxis,:,:]+90)%180-90+VH_phase_per_ant_chan[np.newaxis,np.newaxis,:,:],axis=(0,1))
            new_HV_centering_phase=np.nanmedian(HV_phase_per_ant_chan)
            if new_HV_centering_phase not in convergence:#check for convergence only on HV, assumes VH also converged at same time
                HV_centering_phase=new_HV_centering_phase
                convergence.append(HV_centering_phase)
                print(HV_centering_phase)
            else:
                print('Converged')
                break
        HV_var_per_ant_chan=np.nanvar((anglecalvis[:,:,:,:]-HV_phase_per_ant_chan[np.newaxis,:,np.newaxis,:]+90)%180-90+HV_phase_per_ant_chan[np.newaxis,:,np.newaxis,:],axis=(0,1))
        VH_var_per_ant_chan=np.nanvar((anglecalvis[:,:,:,:]-VH_phase_per_ant_chan[np.newaxis,np.newaxis,:,:]+90)%180-90+VH_phase_per_ant_chan[np.newaxis,np.newaxis,:,:],axis=(0,1))            

    HV_centering_phase_per_chan=np.tile(HV_centering_phase,[len(freqMHz)])
    convergence=[HV_centering_phase]
    for it in range(30):
        wrapped_anglecalvis=(anglecalvis-HV_centering_phase_per_chan[np.newaxis,np.newaxis,np.newaxis,:]+90)%180-90+HV_centering_phase_per_chan[np.newaxis,np.newaxis,np.newaxis,:]
        HV_centering_phase_per_chan=np.nanmedian(wrapped_anglecalvis,axis=(0,1,2))
        new_HV_centering_phase=np.nanmedian(HV_centering_phase_per_chan)
        if new_HV_centering_phase not in convergence:
            HV_centering_phase=new_HV_centering_phase
            convergence.append(HV_centering_phase)
            print(HV_centering_phase)
        else:
            print('Converged')
            break
                    
    if do_plot:        
        fig1=plt.figure(1,figsize=(14,5))
        plt.clf()
        plt.subplot(1,3,1)
        ch=slice(0,1024)
        if do_compute_parallelhands:
            vals=np.angle(cocalvis[:,:,:,ch].reshape(-1))*180/pi
            valid=np.nonzero(isfinite(vals))[0]
            plt.hist(vals[valid],bins=100,label='co products',histtype='step',density=True)
        vals=np.angle(crosscalvis[:,:,:,ch].reshape(-1))*180/pi
        valid=np.nonzero(isfinite(vals))[0]
        plt.hist(vals[valid],bins=100,label='cross products',histtype='step',density=True)
        plt.xlabel('Phase error [deg]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.title('Per polarisation')
        plt.subplot(1,3,2)
        for i in range(8):
            ch=slice(i*1024//8,(i+1)*1024//8)
            vals=np.angle(crosscalvis[:,:,:,ch].reshape(-1))*180/pi
            valid=np.nonzero(isfinite(vals))[0]
            plt.hist(vals[valid],bins=100,label='channels %d-%d'%(ch.start,ch.stop-1),histtype='step',density=True)
        plt.xlabel('Phase error [deg]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.title('Channelised cross hand')
        plt.subplot(1,3,3)
        for i in range(8):
            ch=slice(i*1024//8,(i+1)*1024//8)
            vals=np.angle(crosscalvis[:,:,:,ch].reshape(-1))*180/pi
            vals[np.nonzero(vals<-90)]+=180
            vals[np.nonzero(vals>90)]-=180
            valid=np.nonzero(isfinite(vals))[0]
            plt.hist(vals[valid],bins=100,label='channels %d-%d'%(ch.start,ch.stop-1),histtype='step',density=True)
        plt.xlabel('Phase error [deg]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.title('Channelised cross hand, 180 degree wrapped')
        plt.figtext(0.5,0.96,'Calibration solution error histograms '+'('+filename+' '+f.catalogue.targets[0].aliases[0]+')',ha='center',fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
        fig1.savefig('histogram.pdf')
        
        fig2=plt.figure(2,figsize=(10,4))
        plt.clf()
        if plot_hv_phase_spline is not None:
            plot_hv_phase_spline(freqMHz)
        plt.plot(freqMHz,medianphase,label='median angle')
        plt.plot(freqMHz,phasemedian,label='angle median')
        plt.plot(freqMHz,HV_centering_phase_per_chan,label='wrap 180')
        plt.plot(freqMHz,0.5*(np.nanmedian(HV_phase_per_ant_chan,axis=0)+np.nanmedian(VH_phase_per_ant_chan,axis=0)),label='wrap per ant')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('HV phase [deg]')
        plt.legend()
        plt.grid('both')
        plt.title(filename+' '+f.catalogue.targets[0].aliases[0])
        if freqMHz[0]==544:
            ylim([-30,-5])
        elif freqMHz[0]==856:
            ylim([-50,0])
        else:
            ylim([-190,-170])
        fig2.savefig('hv_phase.pdf')
    
        if True:
            fig3=plt.figure(3,figsize=(10,4))
            plt.plot(freqMHz,HV_centering_phase_per_chan,'.',ms=2,label=filename)
            plt.xlabel('Frequency [MHz]')
            plt.ylabel('HV phase [deg]')
            plt.legend()
            plt.grid('both')
            fig3.savefig('add_hv_phase.pdf')
        if True:
            fig4=plt.figure(4,figsize=(6,4))
            plt.plot(ucocalvis[:,:,:,len(freqMHz)//4].real.reshape(-1),ucocalvis[:,:,:,len(freqMHz)//4].imag.reshape(-1),'.',ms=2,alpha=0.01,label='uncalibrated co-vis')
            plt.plot(cocalvis[:,:,:,len(freqMHz)//4].real.reshape(-1),cocalvis[:,:,:,len(freqMHz)//4].imag.reshape(-1),'.',ms=2,alpha=0.01,label='calibrated co-vis')
            plt.plot(ucrosscalvis[:,:,:,len(freqMHz)//4].real.reshape(-1),ucrosscalvis[:,:,:,len(freqMHz)//4].imag.reshape(-1),'.',ms=2,alpha=0.01,label='uncalibrated cross-vis')
            plt.plot(crosscalvis[:,:,:,len(freqMHz)//4].real.reshape(-1),crosscalvis[:,:,:,len(freqMHz)//4].imag.reshape(-1),'.',ms=2,alpha=0.01,label='calibrated cross-vis')
            plt.axis('equal')
            plt.title(filename+' visibilities at %d MHz'%freqMHz[len(freqMHz)//4])
            custom_lines = [Line2D([0], [0], color=colorcycle[0], lw=4),
                        Line2D([0], [0], color=colorcycle[1], lw=4),
                        Line2D([0], [0], color=colorcycle[2], lw=4),
                        Line2D([0], [0], color=colorcycle[3], lw=4)]
            plt.legend(custom_lines, ['uncalibrated co-vis', 'calibrated co-vis', 'uncalibrated cross-vis', 'calibrated cross-vis'])
            fig4.savefig('visibilities.pdf')
    
    freqMHz1k,medianphase_1k=ensure_1k(freqMHz,medianphase)
    freqMHz1k,phasemedian_1k=ensure_1k(freqMHz,phasemedian)
    freqMHz1k,HV_centering_phase_per_chan_1k=ensure_1k(freqMHz,HV_centering_phase_per_chan)
    freqMHz1k,HV_var_per_ant_chan_1k=ensure_1k(freqMHz,HV_var_per_ant_chan)
    freqMHz1k,VH_var_per_ant_chan_1k=ensure_1k(freqMHz,VH_var_per_ant_chan)
    freqMHz1k,HV_phase_per_ant_chan_1k=ensure_1k(freqMHz,HV_phase_per_ant_chan)
    freqMHz1k,VH_phase_per_ant_chan_1k=ensure_1k(freqMHz,VH_phase_per_ant_chan)
    savedata={'hv_phase_ant':HV_phase_per_ant_chan_1k,'vh_phase_ant':VH_phase_per_ant_chan_1k,'hv_var_ant':HV_var_per_ant_chan_1k,'vh_var_ant':VH_var_per_ant_chan_1k,'hv_phase_wrap':HV_centering_phase_per_chan_1k,'hv_median_phase':medianphase_1k,'hv_phase_median':phasemedian_1k,'freqMHz':freqMHz1k}
    savedata.update(get_info(f))
    np.savez('%s_hv_phase.npz'%basename,**savedata)
        
# itemname='cross_phase','cross_phase_hv','cross_phase_vh'
def plot_hv_phase_results(band='S',minantennas=33,itemname='hv_phase_wrap',receivername=None,targetname=None):
    filenames=glob.glob('*_hv_phase.npz')
    
    fig1=plt.figure(figsize=(10,4))
    hvphaselist=[]
    freqMHzlist=[]
    cbidlist=[]
    paranglist=[]
    receivers={}
    for filename in filenames:
        if not filename[:10].isdigit():
            continue
        fp=np.load(filename)
        for rec in fp['receivers']:
            if rec not in receivers:
                receivers[rec]=1
            else:
                receivers[rec]+=1
        freqMHz=fp['freqMHz']
        cbid=int(filename[:10])
        if receivername is None:
            if itemname is None:
                hvphase=np.nanmedian(0.5*(fp['hv_phase_ant']+fp['vh_phase_ant']),axis=0)
            else:
                hvphase=fp[itemname]
        else:
            if receivername not in fp['receivers']:
                continue
            ind=fp['receivers'].tolist().index(receivername)
            hvphase=0.5*(fp['hv_phase_ant'][ind,:]+fp['vh_phase_ant'][ind,:])
            
        if band[0]=='U':
            if freqMHz[0]!=544:
                continue
            hvphase=(hvphase+180)%360-180
        elif band[0]=='L':
            if freqMHz[0]!=856:
                continue
            hvphase=(hvphase+180)%360-180
        else:
            if freqMHz[0]<1000:
                continue
            hvphase=(hvphase+180+180)%360-180-180
        if targetname in [None,'%s'%fp['targetname']]:#e.g. '3C286'
            print(cbid,fp['targetname'])
        else:
            continue
        print(len(fp['antennas']))
        print(cbid,fp['targetname'],fp['nchannels'])
        if len(fp['antennas'])<minantennas:
            continue
        paranglist.append(fp['parang'])
        cbidlist.append(cbid)
        hvphaselist.append(hvphase)
        freqMHzlist.append(freqMHz)
        # plt.plot(freqMHz,hvphase,'.',ms=2,alpha=0.05,label='%s %s'%(time.ctime(cbid),fp['targetname']))
        if band[0]=='S' and np.nanmean(hvphase)>-90:
            plt.plot(freqMHz,hvphase-180,'.',ms=2,label='%s %s'%(time.ctime(cbid),fp['targetname']))
        elif band[0]=='S' and np.nanmean(hvphase)<-180-90:
            plt.plot(freqMHz,hvphase+360,'.',ms=2,label='%s %s'%(time.ctime(cbid),fp['targetname']))
        else:
            plt.plot(freqMHz,hvphase,'.',ms=2,label='%s %s'%(time.ctime(cbid),fp['targetname']))
    
    if band[0]=='U':
        plot_hv_phase_spline(np.linspace(544,544*2,1024))
    elif band[0]=='L':
        plot_hv_phase_spline(np.linspace(856,856*2,1024))
    if band[0]=='U':
        plt.ylim([-30,-5])
    elif band[0]=='L':
        plt.ylim([-50,-5])
    else:
        plt.ylim([-190,-170])
    plt.grid('both')
    if itemname is None:
        plt.title(receivername)
    else:
        plt.title(itemname)

    fig2=plt.figure(figsize=(10,4))
    fullfreqMHz=sorted(np.unique(freqMHzlist))
    fullhvphase=np.tile(np.nan,[len(hvphaselist),len(fullfreqMHz)])
    for i in range(len(hvphaselist)):
        ind=fullfreqMHz.index(freqMHzlist[i][0])
        fullhvphase[i,ind:ind+1024]=hvphaselist[i]
        
    medianhvphase=np.nanmedian(fullhvphase,axis=0)
    stdhvphase=np.nanstd(fullhvphase,axis=0)
    perchvphase10=np.nanpercentile(fullhvphase,q=[10,90],axis=0)
    perchvphase25=np.nanpercentile(fullhvphase,q=[25,75],axis=0)
    fullfreqMHz=np.array(fullfreqMHz)
        
    plt.fill_between(fullfreqMHz,perchvphase10[0,:],perchvphase10[1,:],facecolor='lightgrey',edgecolor=None,alpha=0.5,label='10-90% percentile')
    plt.fill_between(fullfreqMHz,perchvphase25[0,:],perchvphase25[1,:],facecolor='grey',edgecolor=None,alpha=0.5,label='25-75% percentile')
    plt.plot(fullfreqMHz,medianhvphase,label='median')
    plt.legend()
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('HV phase [deg]')
    if band[0]=='U':
        plot_hv_phase_spline(np.linspace(544,544*2,1024))
    elif band[0]=='L':
        plot_hv_phase_spline(np.linspace(856,856*2,1024))
    if band[0]=='U':
        plt.ylim([-30,-5])
    elif band[0]=='L':
        plt.ylim([-50,-5])
    else:
        plt.ylim([-190,-170])
    plt.grid('both')
    plt.xlim([fullfreqMHz[0],fullfreqMHz[-1]])
    if False:
        np.savez('sband_cross_phase.npz',**{'cross_phase':medianhvphase,'freqMHz':fullfreqMHz,'cross_phase_std':stdhvphase})
        
    if False:#fitting spline
        from scipy.interpolate import BSpline, splrep, splev
        bcross_sky_k = 3
        if freqMHz[0]==856:
            bcross_sky_knots = [856, 856, 856, 856, 963, 1070, 1284, 1525, 1608, 1658, 1711, 1711, 1711, 1711 ] # knots in MHz
            valid=np.nonzero(np.isfinite(HV_centering_phase_per_chan))
            bcross_sky_coefs = splrep(freqMHz[valid], HV_centering_phase_per_chan[valid], xb=bcross_sky_knots[0], xe=bcross_sky_knots[-1], k=bcross_sky_k, t=bcross_sky_knots[bcross_sky_k+1:-bcross_sky_k-1], task=-1)
            spline_interp = splev(freqMHz, bcross_sky_coefs)
            plt.plot(freqMHz,HV_centering_phase_per_chan)
            plt.plot(freqMHz,spline_interp)
        elif freqMHz[0]==544:
            bcross_sky_knots = [544, 544, 544, 544, 563, 571, 578, 612, 680, 816, 870, 899, 914, 952, 971, 986, 1003, 1020, 1050, 1057, 1071, 1088, 1088, 1088, 1088 ]     # knots in MHz
            valid=np.nonzero(np.isfinite(HV_centering_phase_per_chan))
            bcross_sky_coefs = splrep(freqMHz[valid], HV_centering_phase_per_chan[valid], xb=bcross_sky_knots[0], xe=bcross_sky_knots[-1], k=bcross_sky_k, t=bcross_sky_knots[bcross_sky_k+1:-bcross_sky_k-1], task=-1)
            spline_interp = splev(freqMHz, bcross_sky_coefs)
            plt.plot(freqMHz,HV_centering_phase_per_chan)
            plt.plot(freqMHz,spline_interp)
    # valid=np.nonzero(np.isfinite(medianhvphase))
    # weight=1/(perchvphase[1,:]-perchvphase[0,:])
    # weight=1/np.nanstd(fullhvphase,axis=0)
    # bcross_sky_coefs = splrep(fullfreqMHz[valid], medianhvphase[valid], w=weight[valid],xb=1750, xe=3500, k=bcross_sky_k,s=100,task=0)
    # # bcross_sky_coefs = splrep(fullfreqMHz[valid], medianhvphase[valid], xb=1750, xe=3500, k=bcross_sky_k,s=10)
    # spline_interp = splev(fullfreqMHz, bcross_sky_coefs)
    # plt.plot(fullfreqMHz,spline_interp)
    # ylim([-200,-170])
    return receivers


    
parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file> ...]",
                               description="""This processes delay calibration files and uses katsdpcal pipeline
                                           solutions to determine the array average HV phase profile. By default 
                                           figures are produced for inspection when a single file is processed, 
                                           however when multiple files are processed, HV phase results are written
                                           in a batch to disk only. If no files are specified, then the archive is
                                           trawled for all available 1k and 4k delaycal files. """)
parser.add_option("-o", "--output", dest="outfilebase", default=None,
                  help="Base name of output files (*.npz for output data and *.pdf for figures, "
                       "default is '<dataset_name>_cross_phase.npz')")

(opts, args) = parser.parse_args()

if len(args) == 0:#trawl archive for 1k and 4k delaycal files 
    #archive searching adapted from systems-analysis/analysis/katselib.py
    #NOTE: CAS.ProductTransferStatus: SPOOLED means on tape; RECEIVED means in archive
    query="Description: Delaycal AND NumFreqChannels: (1024 OR 4096) AND CAS.ProductTypeName: MeerKATTelescopeProduct AND CAS.ProductTransferStatus: RECEIVED"
    archive = pysolr.Solr('http://kat-archive.kat.ac.za:8983/solr/kat_core')
    result = archive.search(query, sort='CaptureBlockId desc',rows=1000) #print([r.keys() for r in result][0])
    fid2fn = lambda fid: "http://archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)
    # fid2fn = lambda fid: "http://test-archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)
    for r in result:
        if r['MinFreq']<=544e6:
            band='U'
        elif r['MinFreq']<=856e6:
            band='L'
        else:
            band='S'
        # if band[0]!='U':
        #     continue
        cbid=int(r['CaptureBlockId'])
        if len(glob.glob('%d_hv_phase.npz'%cbid)):
            print('Skipping %d'%cbid)
            continue
        filename=fid2fn(cbid)
        print(filename)
        print('Target %s Duration %d NumFreqChannels %d'%(r['Targets'][1],r['Duration'],r['NumFreqChannels']))
        print('DumpPeriod %g MinFreq %g MaxFreq %g'%(r['DumpPeriod'],r['MinFreq'],r['MaxFreq']))
        try:
            analyse_hv_phase(filename,do_compute_parallelhands=False,do_plot=False)
        except KeyboardInterrupt:
            break
        except Exception as e: 
            print('An exception occurred processing %s'%filename)
            print(e)
elif len(args) == 1:
    analyse_hv_phase(filename,do_median_time=False,do_compute_parallelhands=True,do_plot=True,basename=opts.outfilebase)
else:
    # args=[1693662868,1695817735,1693661748,1693657602,1693660786,1693658891,1693496913,1694437059]
    for cbid in args:
        filename=fid2fn(cbid)
        analyse_hv_phase(filename,do_median_time=False,do_compute_parallelhands=False,do_plot=False,basename=opts.outfilebase)
            
if False:#reverse order
    query="Description: Delaycal AND NumFreqChannels: (1024 OR 4096) AND CAS.ProductTypeName: MeerKATTelescopeProduct AND CAS.ProductTransferStatus: RECEIVED"
    archive = pysolr.Solr('http://kat-archive.kat.ac.za:8983/solr/kat_core')
    result = archive.search(query, sort='CaptureBlockId desc',rows=1000) #print([r.keys() for r in result][0])
    fid2fn = lambda fid: "http://archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)
    # fid2fn = lambda fid: "http://test-archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)
    Ucbid=[]
    Lcbid=[]
    Scbid=[]
    for r in result:
        if r['MinFreq']<=544e6:
            Ucbid.append(int(r['CaptureBlockId']))
        elif r['MinFreq']<=856e6:
            Lcbid.append(int(r['CaptureBlockId']))
        else:
            Scbid.append(int(r['CaptureBlockId']))


    for cbid in Lcbid[::-1]:
        if len(glob.glob('%d_hv_phase.npz'%cbid)):
            print('Skipping %d'%cbid)
            continue
        filename=fid2fn(cbid)
        print(filename)
        try:
            analyse_hv_phase(filename,do_compute_parallelhands=False,do_plot=False)
        except KeyboardInterrupt:
            break
        except Exception as e: 
            print('An exception occurred processing %s'%filename)
            print(e)

if False:
    #kim's datasets
    cbids=[1539685313,1534425960,1535892672,1544510348,1548210657,1549681047]
    #m000_rsc_rxl_serial_number on http://portal.mkat.karoo.kat.ac.za/katgui/sensor-graph
    #note m041 is 4066 from 23 jan 2019
    kim_receivers={'m000':'l.4028','m001':'l.4007','m002':'l.4024','m003':'l.4017','m004':'l.4019','m005':'l.4029','m006':'l.4008','m007':'l.4016','m008':'l.4013','m009':'l.4021','m010':'l.4014','m011':'l.4032','m012':'l.4022','m013':'l.4044','m014':'l.4053','m015':'l.4005','m016':'l.4062','m017':'l.4012','m018':'l.4025','m019':'l.4057','m020':'l.4018','m021':'l.4011','m022':'l.4009','m023':'l.4052','m024':'l.4'   ,'m025':'l.4006','m026':'l.4046','m027':'l.4067','m028':'l.4063','m029':'l.4065','m030':'l.4030','m031':'l.4004','m032':'l.4060','m033':'l.4051','m034':'l.4047','m035':'l.4042','m036':'l.4020','m037':'l.4015','m038':'l.4061','m039':'l.4064','m040':'l.4059','m041':'l.4058','m042':'l.4031','m043':'l.4056','m044':'l.4048','m045':'l.4049','m046':'l.4034','m047':'l.4068','m048':'l.4033','m049':'l.4040','m050':'l.4035','m051':'l.4045','m052':'l.4027','m053':'l.4023','m054':'l.4050','m055':'l.4037','m056':'l.4039','m057':'l.4038','m058':'l.4026','m059':'l.4043','m060':'l.4036','m061':'l.4003','m062':'l.4054','m063':'l.4010'}
    cbid=1549681047#does not have cal_product_BCROSS_DIODE0

