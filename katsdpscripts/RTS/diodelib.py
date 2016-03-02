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
import scape

def read_and_plot_data(filename,output_dir='.',pdf=True,Ku = False,verbose = False,error_bars=False,target='off1',write_nd=False):
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

    h5 = katfile.open(filename)
    if verbose: logger.debug(h5.__str__())
    ants = h5.ants
   
    pickle_file = open('/var/kat/katsdpscripts/RTS/rfi_mask.pickle')
    rfi_static_flags = pickle.load(pickle_file)
    pickle_file.close()
    edge = np.tile(True,4096)
    edge[slice(211,3896)] = False
    static_flags = np.logical_or(edge,rfi_static_flags)
    if Ku:
        logger.debug("Using Ku band ... unsetting L band RFI flags")
        h5.spectral_windows[0].centre_freq = 12500.5e6
        # Don't subtract half a channel width as channel 0 is centred on 0 Hz in baseband
        h5.spectral_windows[0].channel_freqs = h5.spectral_windows[0].centre_freq +  h5.spectral_windows[0].channel_width * (np.arange(h5.spectral_windows[0].num_chans) - h5.spectral_windows[0].num_chans / 2)
        static_flags = edge    

    n_ants = len(ants)
    ant_ind = np.arange(n_ants)
    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    pols = ['v','h']
    diode= 'coupler'
    for a,col in zip(ants,colour):
        if pdf: 
            pp = PdfPages(output_dir+'/'+nice_filename+'.'+a.name+'.pdf')
            logger.debug("Created output PDF file: %s"%output_dir+'/'+nice_filename+'.'+a.name+'.pdf')

        if not(Ku): 
            fig1 = plt.figure(2,figsize=(20,5))
        fig2 = plt.figure(1,figsize=(20,5))
        
        fig0 = plt.figure(0,figsize=(20,5))
        h5.select()
        h5.select(ants = a.name,channels=~static_flags)
        d = scape.DataSet(h5)
        scape.plot_xyz(d,'time','amp',label='Average of the data')
        on = h5.sensor['Antennas/'+a.name+'/nd_coupler']
        ts = h5.timestamps - h5.timestamps[0]
        plt.plot(ts,on*4000,'g',label='katdal ND sensor')
        plt.title("Timeseries for antenna %s - %s"%(a.name,git_info()))
        plt.legend()
        for pol in pols:
            logger.debug("Processing: %s%s"%(a.name,pol))
            ant = a.name
            ant_num = int(ant[3])
            
            air_temp = np.mean(h5.sensor['Enviro/air_temperature'])
            if not(Ku):
                diode_filename = '/var/kat/katconfig/user/noise-diode-models/mkat/rx.'+h5.receivers[ant]+'.'+pol+'.csv'
                logger.info('Loading noise diode file %s from config'%diode_filename)
                try:
                    nd = scape.gaincal.NoiseDiodeModel(diode_filename)
                except:
                    logger.error("Error reading the noise diode file ... using a constant value of 20k")
                    logger.error("Be sure to reprocess the data once the file is in the config")
                    nd = scape.gaincal.NoiseDiodeModel(freq=[856,1712],temp=[20,20])
            
            s = h5.spectral_windows[0]
            f_c = s.centre_freq
            #cold data
            logger.debug('Using off target %s'%target)
            h5.select(ants=a.name,pol=pol,channels=~static_flags, targets = target,scans='track')
            freq = h5.channel_freqs
            if not(Ku): nd_temp = nd.temperature(freq / 1e6)
            cold_data = h5.vis[:].real
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            n_on = np.tile(False,on.shape[0])
            n_off = np.tile(False,on.shape[0])
            buff = 5
            if not any(on):
                logger.critical('No noise diode fired during track of %s'%target)
            else:
                jumps = (np.diff(on).nonzero()[0] + 1).tolist()
                n_on[slice(jumps[0]+buff,jumps[1]-buff)] = True
                n_off[slice(jumps[1]+buff,-buff)] = True

            cold_off = n_off
            cold_on = n_on
            #hot data
            h5.select(ants=a.name,pol=pol,channels=~static_flags,targets = 'Moon',scans='track')
            hot_data = h5.vis[:].real
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            n_on = np.tile(False,on.shape[0])
            n_off = np.tile(False,on.shape[0])
            if not any(on):
                logger.critical('No noise diode fired during track of %s'%target)
            else:
                jumps = (np.diff(on).nonzero()[0] + 1).tolist()
                n_on[slice(jumps[0]+buff,jumps[1]-buff)] = True
                n_off[slice(jumps[1]+buff,-buff)] = True

            hot_off = n_off
            hot_on = n_on
            cold_spec = np.mean(cold_data[cold_off,:,0],0)
            hot_spec = np.mean(hot_data[hot_off,:,0],0)
            cold_nd_spec = np.mean(cold_data[cold_on,:,0],0)
            hot_nd_spec = np.mean(hot_data[hot_on,:,0],0)

            if error_bars:
                cold_spec_std = np.std(cold_data[cold_off,:,0],0)
                hot_spec_std = np.std(hot_data[hot_off,:,0],0)
                cold_nd_spec_std = np.std(cold_data[cold_on,:,0],0)
                hot_nd_spec_std = np.std(hot_data[hot_on,:,0],0)
             
            if not(Ku):
                TAh = hot_spec/(hot_nd_spec - hot_spec) * nd_temp # antenna temperature on the moon (from diode calibration)
                TAc = cold_spec/(cold_nd_spec - cold_spec) * nd_temp # antenna temperature on cold sky (from diode calibration) (Tsys)
            Y = hot_spec / cold_spec
            if error_bars: Y_std = Y * np.sqrt((hot_spec_std/hot_spec)**2 + (cold_spec_std/cold_spec)**2)
            D = 13.5
            lam = 3e8/freq
            HPBW = 1.18 *(lam/D)
            Om = 1.133 * HPBW**2  # main beam solid angle for a gaussian beam
            R = np.radians(0.25) # radius of the moon
            Os = np.pi * R**2 # disk source solid angle 
            _f_MHz, _eff_pct = np.loadtxt("/var/kat/katconfig/user/aperture-efficiency/mkat/ant_eff_L_%s_AsBuilt.csv"%pol.upper(), skiprows=2, delimiter="\t", unpack=True)
            eta_A = np.interp(freq,_f_MHz,_eff_pct)/100 # EMSS aperture efficiency
            if Ku: eta_A = 0.7
            Ag = np.pi* (D/2)**2 # Antenna geometric area
            Ae = eta_A * Ag  # Effective aperture
            x = 2*R/HPBW # ratio of source to beam
            K = ((x/1.2)**2) / (1-np.exp(-((x/1.2)**2))) # correction factor for disk source from Baars 1973
            TA_moon = 225 * (Os*Ae/(lam**2)) * (1/K) # contribution from the moon (disk of constant brightness temp)
            if error_bars: Thot_std = 2.25
            gamma = 1.0
            if error_bars: gamma_std = 0.01
            Tsys = gamma * (TA_moon)/(Y-gamma) # Tsys from y-method ... compare with diode TAc
            if error_bars: Tsys_std = Tsys * np.sqrt((Thot_std/Thot)**2 + (Y_std/Y)**2 + (gamma_std/gamma)**2)
            if not(Ku):
                Ydiode = hot_nd_spec / hot_spec
                Tdiode = (TA_moon + Tsys)*(Ydiode/gamma-1)
            
            p = 1 if pol == 'v' else 2
            if not(Ku):
                plt.figure(2)
                plt.subplot(1,2,p)
                plt.ylim(0,50)
                plt.ylabel('T_ND [K]')
                plt.xlim(900,1670)
                plt.xlabel('f [MHz]')
                if p ==ant_num * 2-1: plt.ylabel(ant)
                plt.axhspan(14, 35, facecolor='g', alpha=0.5)
                plt.plot(freq/1e6,Tdiode,'b.',label='Measurement: Y-method')
                if write_nd:
                    outfilename = '%s/%s.%s.%s.%s.csv' % (output_dir,ant, diode, pol.lower(),file_base)
                    outfile = file(outfilename, 'w')
                    outfile.write('#Data from %s\n# Frequency [Hz], Temperature [K]\n'%file_base)
                    # Write CSV part of file
                    outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(freq,medfilt(Tdiode))]))
                    outfile.close()
                    logger.info('Noise temp data written to file %s'%outfilename)
                plt.plot(freq/1e6,nd_temp,'k.',label='Model: EMSS')
                plt.grid()
                plt.legend()
                
            plt.figure(1)
            plt.subplot(1,2,p)
            if not(Ku): plt.ylim(15,50)
            plt.ylabel('Tsys/eta_A [K]')
            if not(Ku): plt.xlim(900,1670)
            plt.xlabel('f [MHz]')
            if p == ant_num * 2 -1: plt.ylabel(ant)
            if error_bars: plt.errorbar(freq/1e6,Tsys,Tsys_std,color = 'b',linestyle = '.',label='Measurement')
            plt.plot(freq/1e6,Tsys/eta_A,'b.',label='Measurement: Y-method')
            if not(Ku): plt.plot(freq/1e6,TAc/eta_A,'c.',label='Measurement: ND calibration')
            plt.axhline(np.mean(Tsys/eta_A),linewidth=2,color='k',label='Mean: Y-method')
            spec_Tsys_eta = 0*freq
            spec_Tsys_eta[freq<1420e6] =  42 # [R.T.P095] == 220
            spec_Tsys_eta[freq>=1420e6] =  46 # [R.T.P.096] == 200
            if not(Ku): plt.plot(freq/1e6, spec_Tsys_eta,'r',linewidth=2,label='PDR Spec')
            if not(Ku): plt.plot(freq/1e6,np.interp(freq/1e6,[900,1670],[(64*Ag)/275.0,(64*Ag)/410.0]),'g',linewidth=2,label="275-410 m^2/K at Receivers CDR")

            plt.grid()
            plt.legend(loc=2,fontsize=12)
        
        if not(Ku):
            plt.figure(2)
            plt.subplot(1,2,1)
            ax = plt.gca()
            ax.text(0.95, 0.01,git_info(), horizontalalignment='right',fontsize=10,transform=ax.transAxes)
            plt.title('%s Coupler Diode: V pol: %s'%(ant,file_base))
            plt.subplot(1,2,2)
            ax = plt.gca()
            ax.text(0.95, 0.01,git_info(), horizontalalignment='right',fontsize=10,transform=ax.transAxes)
            plt.title('%s Coupler Diode: H pol: %s'%(ant,file_base))

        plt.figure(1)
        plt.subplot(1,2,1)
        ax = plt.gca()
        ax.text(0.95, 0.01,git_info(), horizontalalignment='right',fontsize=10,transform=ax.transAxes)
        plt.title('%s Tsys/eta_A: V pol: %s'%(ant,file_base))
        plt.subplot(1,2,2)
        ax = plt.gca()
        ax.text(0.95, 0.01,git_info(), horizontalalignment='right',fontsize=10,transform=ax.transAxes)
        plt.title('%s Tsys/eta_A: H pol: %s'%(ant,file_base))
        if pdf:
            if not(Ku):
                fig1.savefig(pp,format='pdf')
            fig2.savefig(pp,format='pdf')
            fig0.savefig(pp,format='pdf')
            pp.close() # close the pdf file
            plt.close("all")
    logger.info('Processing complete')



# test main method for the library
if __name__ == "__main__":
#test the method with a know file
    filename = '/var/kat/archive/data/RTS/telescope_products/2015/04/22/1429706275.h5'
    pdf=True
    Ku=False
    verbose=False
    out = '.'
    error_bars = False
    target = 'off1'
    write_nd = False
    print 'Performing test run with: ' + filename
    read_and_plot_data(filename,out,pdf,Ku,verbose,error_bars,target,write_nd)
