#!/usr/bin/python
# mattieu@sarao.ac.za 19 September 2023
# Script that uses katsdpcal's delay calibration visibilties and calibration solutions to measure the HV phase.
# This method is an implementation of a description of how Kim McAlpine determined the original L-band BSpline
# that is routinely used in the CAL pipeline. The method has been extended to work on unpolarised targets, and
# in other bands.

import optparse
import katdal
import pysolr
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, splev

try:
    from katholog.utilities import plot_hv_phase_spline
except:
    plot_hv_phase_spline=None
    pass

#the channelisation might be 1K,4K,32K 
def ensure_1k(freqMHz,xyphase):
    if len(freqMHz)==1024:
        return freqMHz,xyphase
    elif len(freqMHz)==1024*4:
        return freqMHz[::4],np.nanmedian(xyphase.reshape([1024,4]),axis=1)
    elif len(freqMHz)==1024*32:
        return freqMHz[::32],np.nanmedian(xyphase.reshape([1024,32]),axis=1)        
    raise RuntimeError('Unexpected number of channels: %d'%len(freqMHz))
    
# filename='1693662868_sdp_l0.full.rdb'#lband
# filename='1693661748_sdp_l0.full.rdb'#uhf
# filename='./s0/1693657602_sdp_l0.full.rdb'#s0
# filename='./s2/1693660786_sdp_l0.full.rdb'#s2
# filename='./s4/1693658891_sdp_l0.full.rdb'#s4
# filename='1694437059_sdp_l0.full.rdb'#l-band pks1934
#do_median_diode_antennas=True#take median over antenna for noise diode corrections (default required behaviour)
#do_median_time=False#take median over time of uncorrected visibilities before processing
#do_compute_parallelhands=True #for displaying parallel hand histograms (slower)
def analyse_hv_phase(filename,do_median_diode_antennas=True,do_median_time=False,do_compute_parallelhands=True,do_plot=True,basename=None):
    f=katdal.open(filename)
    if basename is None:
        basename=f.source.capture_block_id
    if do_compute_parallelhands:    
        f.select(corrprods='cross', scans = 1) #select un_corrected track data, but not slew, not stop (antenna is not tracking anymore while cal is doing calculations)
    else:
        f.select(corrprods='cross', pol='hv,vh', scans = 1)
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
        print('Unexpected: cal_product_G event 0 is expected to be nan, event 1 is expected to be valid')
        return
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
    calvis=np.tile(np.nan+1j*np.nan,[4,ndumps,len(f.ants),len(f.ants),len(freqMHz)])
    for iant in range(len(f.ants)):
        for jant in range(iant+1,len(f.ants)):
            for iorder,polorder in enumerate([[1,1],[1,0],[0,1],[0,0]]):#hh, hv, vh, vv
                if not do_compute_parallelhands:
                    if iorder in [0,3]:
                        continue                    
                ipol,jpol=polorder#0 is v, 1 is h
                product=[f.ants[iant].name+cal_pol_ordering[ipol],f.ants[jant].name+cal_pol_ordering[jpol]]
                K_delay_ns=(f.sensor.get('cal_product_K')[idump_K][ipol,iant]-f.sensor.get('cal_product_K')[idump_K][jpol,jant])*1e9
                if do_median_diode_antennas:
                    KCROSS_DIODE_delay_ns=(np.nanmedian(f.sensor.get('cal_product_KCROSS_DIODE')[idump_KX][ipol,:],axis=-1)-np.nanmedian(f.sensor.get('cal_product_KCROSS_DIODE')[idump_KX][jpol,:],axis=-1))*1e9
                else:
                    KCROSS_DIODE_delay_ns=(f.sensor.get('cal_product_KCROSS_DIODE')[idump_KX][ipol,iant]-f.sensor.get('cal_product_KCROSS_DIODE')[idump_KX][jpol,jant])*1e9
                K_phase_rad=2*np.pi*freqMHz*1e6*K_delay_ns/1e9
                K=np.exp(1j*K_phase_rad)
                KCROSS_DIODE_phase_rad=2*np.pi*freqMHz*1e6*KCROSS_DIODE_delay_ns/1e9
                KCROSS_DIODE=np.exp(1j*KCROSS_DIODE_phase_rad)
                B=(f.sensor.get('cal_product_B0')[idump_B][:,ipol,iant])*np.conj(f.sensor.get('cal_product_B0')[idump_B][:,jpol,jant])
                G=f.sensor.get('cal_product_G')[idump_G][ipol,iant]*np.conj(f.sensor.get('cal_product_G')[idump_G][jpol,jant])
                if do_median_diode_antennas:
                    BCROSS_DIODE=(np.nanmedian(f.sensor.get('cal_product_BCROSS_DIODE0')[idump_BX][:,ipol,:],axis=-1))*np.conj(np.nanmedian(f.sensor.get('cal_product_BCROSS_DIODE0')[idump_BX][:,jpol,:],axis=-1))
                else:
                    BCROSS_DIODE=(f.sensor.get('cal_product_BCROSS_DIODE0')[idump_BX][:,ipol,iant])*np.conj(f.sensor.get('cal_product_BCROSS_DIODE0')[idump_BX][:,jpol,jant])
                productindex=corr_prod_list.index(product)
                correctedvis=uncorrectedvis[:,:,productindex]/(G*K*B*KCROSS_DIODE*BCROSS_DIODE)[np.newaxis,:]
                if iorder==1:#conjugates the HV visibilities
                    calvis[iorder,:,iant,jant,:]=np.exp(-1j*np.angle(correctedvis))
                else:
                    calvis[iorder,:,iant,jant,:]=np.exp(1j*np.angle(correctedvis))                        
    
    #determine average HVphase estimate as center starting point for wrapping
    #only works properly if target is polarised
    if f.catalogue.targets[0].aliases[0] =='3C286': 
        hist_counts=[]
        for i in range(8):
            ch=slice(i*1024//8,(i+1)*1024//8)
            vals=np.angle(calvis[1:3,:,:,:,ch].reshape(-1))*180/pi
            valid=np.nonzero(isfinite(vals))[0]
            hist_count,bin_edge=histogram(vals[valid],bins=100,density=True)
            hist_counts.append(hist_count)
        ipeak_chanrange=np.argmax(np.max(hist_counts,axis=1))
        ipeak_hvphase_offset=np.argmax(np.max(hist_counts,axis=0))
        HV_centering_phase=bin_edge[ipeak_hvphase_offset]
        if HV_centering_phase>90:#in S-band, this might be near -180 or 180; lets force to -180
            HV_centering_phase-=-360# for consistency across datasets and subbands
    else:#unpolarised target - cannot distinguish 180 degree ambiguity, use prior knowledge starting poing
        if freqMHz[0]<800:#UHF
            HV_centering_phase=-14
        elif freqMHz[0]<1000:#Lband
            HV_centering_phase=-23.5
        else:#S-band
            HV_centering_phase=-180
    HV_centering_phase_per_chan=np.tile(HV_centering_phase,[len(freqMHz)])

    anglecalvis=np.angle(calvis[1:3,:,:,:,:])*180/pi
    convergence=[HV_centering_phase]
    for it in range(30):
        wrapped_anglecalvis=(anglecalvis-HV_centering_phase_per_chan[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]+90)%180-90+HV_centering_phase_per_chan[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]
        HV_centering_phase_per_chan=np.nanmedian(wrapped_anglecalvis,axis=(0,1,2,3))
        new_HV_centering_phase=np.nanmedian(HV_centering_phase_per_chan)
        if new_HV_centering_phase in convergence:
            print('Converged')
        else:
            HV_centering_phase=new_HV_centering_phase
            print(HV_centering_phase)
    if do_plot:
        medianphase=np.angle(np.nanmedian(calvis.real[1:3],axis=(0,1,2,3))+1j*np.nanmedian(calvis.imag[1:3],axis=(0,1,2,3)))*180/pi
        fig1=plt.figure(1,figsize=(14,5))
        plt.clf()
        plt.subplot(1,3,1)
        ch=slice(0,1024)
        polnames=['HH','HV*','VH','VV']
        for ipol in range(4):
            vals=np.angle(calvis[ipol,:,:,:,ch].reshape(-1))*180/pi
            valid=np.nonzero(isfinite(vals))[0]
            plt.hist(vals[valid],bins=100,label=polnames[ipol]+' products',histtype='step',density=True)
        plt.xlabel('Phase error [deg]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.title('Per polarisation')
        plt.subplot(1,3,2)
        for i in range(8):
            ch=slice(i*1024//8,(i+1)*1024//8)
            vals=np.angle(calvis[1:3,:,:,:,ch].reshape(-1))*180/pi
            valid=np.nonzero(isfinite(vals))[0]
            plt.hist(vals[valid],bins=100,label='channels %d-%d'%(ch.start,ch.stop-1),histtype='step',density=True)
        plt.xlabel('Phase error [deg]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.title('Channelised cross hand')
        plt.subplot(1,3,3)
        for i in range(8):
            ch=slice(i*1024//8,(i+1)*1024//8)
            vals=np.angle(calvis[1:3,:,:,:,ch].reshape(-1))*180/pi
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
        plt.plot(freqMHz,medianphase,'.',ms=2,label='median')
        plt.plot(freqMHz,np.nanmedian(wrapped_anglecalvis,axis=(0,1,2,3)),'.',ms=2,label='wrap 180')
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
    
        if False:
            fig3=plt.figure(3,figsize=(10,4))
            plt.plot(freqMHz,HV_centering_phase_per_chan,'.',ms=2,label=filename)
            plt.xlabel('Frequency [MHz]')
            plt.ylabel('HV phase [deg]')
            plt.legend()
            plt.grid('both')
            fig3.savefig('add_hv_phase.pdf')
    
    freqMHz1k,HV_centering_phase_per_chan_1k=ensure_1k(freqMHz,HV_centering_phase_per_chan)
    np.savez('%s_hv_phase.npz'%basename,**{'hv_phase':HV_centering_phase_per_chan_1k,'freqMHz':freqMHz1k,'elevation':np.mean(f.el),'targetname':f.catalogue.targets[0].aliases[0]})
    
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


parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file> ...]",
                               description="""This processes delay calibration files and uses katsdpcal pipeline
                                           solutions to determine the array average HV phase profile. By default 
                                           figures are produced for inspection when a single file is processed, 
                                           however when multiple files are processed, HV phase results are written
                                           in a batch to disk only.""")
parser.add_option("-o", "--output", dest="outfilebase", default=None,
                  help="Base name of output files (*.npz for output data and *.pdf for figures, "
                       "default is '<dataset_name>_hv_phase.npz')")

(opts, args) = parser.parse_args()

if len(args) == 0:
    #NOTE: CAS.ProductTransferStatus: SPOOLED means on tape; RECEIVED means in archive
    query="Description: Delaycal AND NumFreqChannels: (1024 OR 4096) AND CAS.ProductTypeName: MeerKATTelescopeProduct AND CAS.ProductTransferStatus: RECEIVED"
    archive = pysolr.Solr('http://kat-archive.kat.ac.za:8983/solr/kat_core')
    result = archive.search(query, sort='CaptureBlockId desc',rows=1000)
    # [r.keys() for r in result]
    fid2fn = lambda fid: "http://archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)
    # fid2fn = lambda fid: "http://test-archive-gw-1.kat.ac.za/%d/%d_sdp_l0.full.rdb"%(fid,fid)
    for r in result:
        cbid=int(r['CaptureBlockId'])
        if len(glob.glob('%d_hv_phase.npz'%cbid)):
            print('Skipping %d'%cbid)
            continue
        filename=fid2fn(cbid)
        print(filename)
        print('Target %s Duration %d NumFreqChannels %d'%(r['Targets'][1],r['Duration'],r['NumFreqChannels']))
        print('DumpPeriod %g MinFreq %g MaxFreq %g'%(r['DumpPeriod'],r['MinFreq'],r['MaxFreq']))
        analyse_hv_phase(filename,do_compute_parallelhands=False,do_plot=False)
elif len(args) == 1:
    analyse_hv_phase(filename,do_median_diode_antennas=True,do_median_time=False,do_compute_parallelhands=True,do_plot=True,basename=opts.outfilebase)
else:
    for filename in args:
        analyse_hv_phase(filename,do_median_diode_antennas=True,do_median_time=False,do_compute_parallelhands=False,do_plot=False,basename=opts.outfilebase)


