#!/usr/bin/python
import katdal as katfile
import scape
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle
from katsdpscripts import git_info
from scipy.signal import medfilt
import logging
import scikits.fitting as fit
import time, ephem


def save_ND(diode_filename,file_base,freq,Tdiode_pol ):
    outfilename = diode_filename.split('/')[-1]
    outfile = file(outfilename, 'w')
    outfile.write('#Data from %s\n# Frequency [Hz], Temperature [K]\n'%file_base)
    # Write CSV part of file
    outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(freq,medfilt(Tdiode_pol))]))
    outfile.close()
    return outfilename
    

def get_nd_on_off(h5,buff = 2,log=None): 
    on = h5.sensor['Antennas/%s/nd_coupler'%(h5.ants[0].name)]
    off = ~h5.sensor['Antennas/%s/nd_coupler'%(h5.ants[0].name)]
    n_on = np.tile(False,on.shape[0])
    n_off = np.tile(False,on.shape[0])
    if not any(on):
        if log is not None:
            log.critical('No noise diode fired during track of %s'%target)
        else :
            print(('No noise diode fired during track of %s'%target))
    else:
        #jumps = (np.diff(on).nonzero()[0] + 1).tolist()
        n_on[on.nonzero()[0][buff:-buff]]   = True
        n_off[off.nonzero()[0][buff:-buff]] = True
    return n_on,n_off


def plot_Tsys_eta_A(freq,Tsys,eta_A,TAc,Ku=False,Tsys_std=None,ant = '', file_base='.'):
    fig = plt.figure(figsize=(20,5))
    pols = ['v','h']
    for p,pol in enumerate(pols) : 
        fig.add_subplot(1,2,p+1)  
        ax = plt.gca()
        ax.text(0.95, 0.01,git_info(), horizontalalignment='right',fontsize=10,transform=ax.transAxes)
        plt.title('%s $T_{sys}/eta_{A}$: %s pol: %s'%(ant,str(pol).upper(),file_base))
        plt.ylabel("$T_{sys}/eta_{A}$ [K]")
        plt.xlabel('f [MHz]')
        #if p == ant_num * 2 -1: plt.ylabel(ant)
        if Tsys_std[pol] is not None :
            plt.errorbar(freq/1e6,Tsys[pol],Tsys_std[pol],color = 'b',linestyle = '.',label='Measurement')
        plt.plot(freq/1e6,Tsys[pol]/eta_A[pol],'b.',label='Measurement: Y-method')
        if not(Ku): plt.plot(freq/1e6,TAc[pol]/eta_A[pol],'c.',label='Measurement: ND calibration')
        plt.axhline(np.mean(Tsys[pol]/eta_A[pol]),linewidth=2,color='k',label='Mean: Y-method')
        if freq.min() < 2090e6:
            D = 13.5
            Ag = np.pi* (D/2)**2 # Antenna geometric area
            spec_Tsys_eta = np.zeros_like(freq)
            plt.ylim(15,50)
            #plt.xlim(900,1670)
            spec_Tsys_eta[freq<1420e6] =  42 # [R.T.P095] == 220
            spec_Tsys_eta[freq>=1420e6] =  46 # [R.T.P.096] == 200
            plt.plot(freq/1e6, spec_Tsys_eta,'r',linewidth=2,label='PDR Spec')
            plt.plot(freq/1e6,np.interp(freq/1e6,[900,1670],[(64*Ag)/275.0,
                    (64*Ag)/410.0]),'g',linewidth=2,label="275-410 m^2/K at Receivers CDR")
            plt.grid()
            plt.legend(loc=2,fontsize=12)
            
    return fig   

def plot_nd(freq,Tdiode,nd_temp,ant = '', file_base=''): 
    fig = plt.figure(figsize=(20,5))
    pols = ['v','h']
    for p,pol in enumerate(pols) : 
        fig.add_subplot(1,2,p+1) # 
        ax = plt.gca()
        ax.text(0.95, 0.01,git_info(), horizontalalignment='right',fontsize=10,transform=ax.transAxes)
        plt.title('%s Coupler Diode: %s pol: %s'%(ant,str(pol).upper(),file_base))
        plt.ylim(0,50)
        plt.ylabel('$T_{ND}$ [K]')
        #plt.xlim(900,1670)
        plt.xlabel('f [MHz]')
        #plt.ylabel(ant)
        plt.axhspan(14, 35, facecolor='g', alpha=0.5)
        plt.plot(freq/1e6,Tdiode[pol],'b.',label='Measurement: Y-method')
        plt.plot(freq/1e6,nd_temp[pol],'k.',label='Model: EMSS')
        plt.grid()
        plt.legend()
    return fig   

def plot_ts(h5,on_ts=None):
    import scape
    fig = plt.figure(figsize=(20,5))
    a = h5.ants[0]
    nd = scape.gaincal.NoiseDiodeModel(freq=[856,1712],temp=[20,20])
    d = scape.DataSet(h5,nd_h_model = nd, nd_v_model=nd )
    scape.plot_xyz(d,'time','amp',label='Average of the data')
    if on_ts is not None :
        on = on_ts
    else :
        on = h5.sensor['Antennas/'+a.name+'/nd_coupler']
    ts = h5.timestamps - h5.timestamps[0]
    plt.plot(ts,np.array(on).astype(float)*4000,'g',label='katdal ND sensor')
    plt.title("Timeseries for antenna %s - %s"%(a.name,git_info()))
    plt.legend()
    return fig

def get_fit(x,y,order=0,true_equals=None):
    """true_equals is a  test that returns bool values to be fitted"""
    if true_equals is None :
        y = y.astype(float)
    else :
        y =   y==true_equals
    return fit.PiecewisePolynomial1DFit(order).fit(x,y)#(timestamps[:])

def Dmoon(observer):
    """ The moon's apparante diameter changes by ~10% during over a year, and
        an x% error in Dmoon leads to systemtic error in Thot scaling as 2*x/lambda!
       @param observer: either an ephem.Observer or a suitable date (then at earth's centre).
       @return [rad] diameter of the moon, from the earth on the specified date (ephem.Observer)"""
    observer = observer if isinstance(observer,ephem.Observer) else ephem.Date(observer)
    moon = ephem.Moon(observer)
    D = moon.size/3600. # deg
    return D*np.pi/180. # rad


def read_and_plot_data(filename,output_dir='.',pdf=True,Ku = False,
                        verbose = False,error_bars=False,target='off1',
                        write_nd=False,rfi_mask='/var/kat/katsdpscripts/RTS/rfi_mask.pickle',**kwargs):
    print('inside',kwargs)
    file_base = filename.split('/')[-1].split('.')[0]
    nice_filename =  file_base + '_T_sys_T_nd'
    
    # Set up logging: logging everything (DEBUG & above), both to console and file
    logger = logging.root
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(nice_filename + '.log', 'w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(fh)
    logger.info('Beginning data processing with:\n%s'%git_info('standard'))
    
    if Ku:
        logger.debug("Using Ku band ... unsetting L band RFI flags")
        h5 = katfile.open(filename,centre_freq = 12500.5e6 , **kwargs)
        length = h5.shape[1]
        # Don't subtract half a channel width as channel 0 is centred on 0 Hz in baseband
        rfi_static_flags = np.tile(True,length)
    else :
        h5 = katfile.open(filename,**kwargs)
        length = h5.shape[1]
        pickle_file = open(rfi_mask,mode='rb')
        rfi_static_flags = pickle.load(pickle_file)
        pickle_file.close()
        # Now find the edges of the mask
        rfi_freqs_width = (856000000.0/rfi_static_flags.shape[0])
        rfi_freqs_min = 856000000.0-rfi_freqs_width/2. # True Start of bin
        rfi_freqs_max = rfi_freqs_min*2-rfi_freqs_width/2.  # Middle of Max-1 bin
        rfi_freqs = np.linspace(rfi_freqs_min,rfi_freqs_max,rfi_static_flags.shape[0]) 
        rfi_function  = get_fit(np.r_[rfi_freqs,rfi_freqs+rfi_freqs_width*0.9999] , np.r_[rfi_static_flags,rfi_static_flags])
        rfi_static_flags = rfi_function(h5.channel_freqs)
        #print ( (rfi_static_flags_new-rfi_static_flags)**2).sum()
    if verbose: logger.debug(h5.__str__())
    edge = np.tile(True,length)
    #edge[slice(211,3896)] = False #Old
    edge[slice(int(round(length*0.0515)),int(round(0.9512*length)))] = False###
    static_flags = np.logical_or(edge,rfi_static_flags)
    
    ants = h5.ants

   
    
    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    pols = ['v','h']
    
    for a in ants:
        ant = a.name
        try:
            rx_sn = h5.receivers[ant]
        except KeyError:
            logger.error('Receiver serial number for antennna %s not found in the H5 file'%ant)
            rx_sn = 'l.SN_NOT_FOUND'
        band,SN = rx_sn.split('.')
        if pdf:
            pdf_filename = output_dir+'/'+nice_filename+'.'+rx_sn+'.'+a.name+'.pdf'
            pp = PdfPages(pdf_filename)
            logger.debug("Created output PDF file: %s"%pdf_filename)

        #fig0 = plt.figure(0,figsize=(20,5))
        h5.select()
        h5.select(ants = a.name,channels=~static_flags)
        observer = h5.ants[0].observer; observer.date = time.gmtime(h5.timestamps.mean())[:6] # katdal resets this date to now()!
        fig0 = plot_ts(h5)
        Tsys, TAc, Tsys_std = {}, {}, {}
        eta_A = {}
        Tdiode = {}
        nd_temp = {}
        for pol in pols:
            logger.debug("Processing: %s%s"%(a.name,pol))
            Tsys_std[pol] = None
            if not(Ku):
                diode_filename = '/var/kat/katconfig/user/noise-diode-models/mkat/rx.'+rx_sn+'.'+pol+'.csv'
                logger.info('Loading noise diode file %s from config'%diode_filename)
                try:
                    nd = scape.gaincal.NoiseDiodeModel(diode_filename)
                except:
                    logger.error("Error reading the noise diode file ... using a constant value of 20k")
                    logger.error("Be sure to reprocess the data once the file is in the config")
                    nd = scape.gaincal.NoiseDiodeModel(freq=[856,1712],temp=[20,20])
            
            #cold data
            logger.debug('Using off target %s'%target)
            h5.select(ants=a.name,pol=pol,channels=~static_flags, targets = target,scans='track')
            freq = h5.channel_freqs
            
            cold_data = h5.vis[:].real
            cold_on,cold_off = get_nd_on_off(h5,log=logger)
            #hot data
            h5.select(ants=a.name,pol=pol,channels=~static_flags,targets = 'Moon',scans='track')
            hot_on,hot_off = get_nd_on_off(h5,log=logger)
            hot_data = h5.vis[:].real

            cold_spec = np.mean(cold_data[cold_off,:,0],0)
            hot_spec = np.mean(hot_data[hot_off,:,0],0)
            cold_nd_spec = np.mean(cold_data[cold_on,:,0],0)
            hot_nd_spec = np.mean(hot_data[hot_on,:,0],0)

            if not(Ku):
                nd_temp[pol] = nd.temperature(freq / 1e6)
                # antenna temperature on the moon (from diode calibration)
                TAh = hot_spec/(hot_nd_spec - hot_spec) * nd_temp[pol] 
                # antenna temperature on cold sky (from diode calibration) (Tsys)
                TAc[pol] = cold_spec/(cold_nd_spec - cold_spec) * nd_temp[pol] 
                print(("Mean TAh = %f  mean TAc = %f "%(TAh.mean(),TAc[pol].mean())))
            Y = hot_spec / cold_spec
            D = 13.5 # Efficiency tables are defined for 13.5
            lam = 299792458./freq
            HPBW = 1.18 *(lam/D)
            Om = 1.133 * HPBW**2  # main beam solid angle for a gaussian beam
            R = 0.5*Dmoon(observer) # radius of the moon
            Os = 2*np.pi*(1-np.cos(R)) # disk source solid angle 
            _f_MHz, _eff_pct = np.loadtxt("/var/kat/katconfig/user/aperture-efficiency/mkat/ant_eff_%s_%s_AsBuilt.csv"%(band.upper(),pol.upper()), skiprows=2, delimiter="\t", unpack=True)
            eta_A[pol] = np.interp(freq,_f_MHz,_eff_pct)/100. # EMSS aperture efficiency
            if Ku: eta_A[pol] = 0.7
            Ag = np.pi* (D/2)**2 # Antenna geometric area
            Ae = eta_A[pol] * Ag  # Effective aperture
            x = 2*R/HPBW # ratio of source to beam
            K = ((x/1.2)**2) / (1-np.exp(-((x/1.2)**2))) # correction factor for disk source from Baars 1973
            TA_moon = 225 * (Os/Om) * (1/K) # contribution from the moon (disk of constant brightness temp)
            gamma = 1.0
            Thot = TA_moon
            Tcold = 0
            Tsys[pol] = gamma * (Thot-Tcold)/(Y-gamma) # Tsys from y-method ... compare with diode TAc
            if error_bars:
                cold_spec_std = np.std(cold_data[cold_off,:,0],0)
                hot_spec_std = np.std(hot_data[hot_off,:,0],0)
                cold_nd_spec_std = np.std(cold_data[cold_on,:,0],0)
                hot_nd_spec_std = np.std(hot_data[hot_on,:,0],0)
                Y_std = Y * np.sqrt((hot_spec_std/hot_spec)**2 + (cold_spec_std/cold_spec)**2)
                Thot_std = 2.25
                gamma_std = 0.01
                 # This is not definded
                raise NotImplementedError("The factor Thot  has not been defined ")
                Tsys_std[pol] = Tsys[pol] * np.sqrt((Thot_std/Thot)**2 + (Y_std/Y)**2 + (gamma_std/gamma)**2)
            else :
                Tsys_std[pol] = None
            if not(Ku):
                Ydiode = hot_nd_spec / hot_spec
                Tdiode_h = (Tsys[pol]-Tcold+Thot)*(Ydiode/gamma-1) # Tsys as computed above includes Tcold
                Ydiode = cold_nd_spec / cold_spec
                Tdiode_c = Tsys[pol]*(Ydiode/gamma-1)
                Tdiode[pol] = (Tdiode_h+Tdiode_c)/2. # Average two equally valid, independent results
            if write_nd:
                outfilename = save_ND(diode_filename,file_base,freq,Tdiode[pol] )
                logger.info('Noise temp data written to file %s'%outfilename)
        
        fig2 = plot_nd(freq,Tdiode,nd_temp,ant = ant, file_base=file_base)
        fig1 = plot_Tsys_eta_A(freq,Tsys,eta_A,TAc,Tsys_std=Tsys_std,ant = ant, file_base=file_base,Ku=Ku)
        if pdf:
            if not(Ku):
                fig2.savefig(pp,format='pdf')
            fig1.savefig(pp,format='pdf')
            fig0.savefig(pp,format='pdf')
            pp.close() # close the pdf file
            plt.close("all")
    logger.info('Processing complete')



# test main method for the library
if __name__ == "__main__":
#test the method with a know file
    filename = '/var/kat/archive/data/RTS/telescope_products/2016/01/19/1453216690.h5'
    #filename = '/var/kat/archive2/data/MeerKATAR1/telescope_products/2016/07/21/1469134098.h5'
    pdf=True
    Ku=False
    verbose=False
    out = '.'
    error_bars = False
    target = 'off1'
    write_nd = True
    print('Performing test run with: ' + filename)
    read_and_plot_data(filename,out,pdf,Ku,verbose,error_bars,target,write_nd)
# Output checsums of the noisediode files for 1453216690.h5s
#md5sum rx.* 
#7545907bbe621f4e5937de7536be21f3  rx.l.4002.h.csv
#532332722c8bca9714f4e2d97978ccd6  rx.l.4002.v.csv
#66678871aeaadc4178e4420bef9861aa  rx.l.4.h.csv
#af113bd4ab10bb61386521f61eb4d716  rx.l.4.v.csv

