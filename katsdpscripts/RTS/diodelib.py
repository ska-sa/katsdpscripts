#!/usr/bin/python
import katdal as katfile
import scape
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

def read_and_plot_data(filename,output_dir='.',pdf=True,Ku = False,verbose = False,error_bars=False):
    file_base = filename.split('/')[-1].split('.')[0]
    nice_filename =  file_base + '_T_sys_T_nd'
    if pdf: pp = PdfPages(output_dir+'/'+nice_filename+'.pdf')

    h5 = katfile.open(filename)
    if verbose: print h5 
    
    pickle_file = open('/var/kat/katsdpscripts/RTS/rfi_mask.pickle')
    rfi_static_flags = pickle.load(pickle_file)
    pickle_file.close()
    edge = np.tile(True,4096)
    edge[slice(211,3896)] = False
    static_flags = np.logical_or(edge,rfi_static_flags)
    if Ku:
        h5.spectral_windows[0].centre_freq = 12500.5e6
        # Don't subtract half a channel width as channel 0 is centred on 0 Hz in baseband
        h5.spectral_windows[0].channel_freqs = h5.spectral_windows[0].centre_freq +  h5.spectral_windows[0].channel_width * (np.arange(h5.spectral_windows[0].num_chans) - h5.spectral_windows[0].num_chans / 2)
        static_flags = edge    

    ants = h5.ants
    n_ants = len(ants)

    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    pols = ['v','h']
    diode= 'coupler'
    if not(Ku): 
        fig1 = plt.figure(1,figsize = (20,5))
    fig2 = plt.figure(2,figsize = (20,5))
    rx_serial = str(4)
    rx_band = 'l'
    for pol in pols:
        for a,col in zip(ants,colour):    
            ant = a.name
            ant_num = int(ant[3])
            air_temp = np.mean(h5.sensor['Enviro/air_temperature'])
            if not(Ku):
                diode_filename = '/var/kat/katconfig/user/noise-diode-models/mkat/rx.'+rx_band+'.'+rx_serial+'.'+pol+'.csv'
                nd = scape.gaincal.NoiseDiodeModel(diode_filename)
            
            s = h5.spectral_windows[0]
            f_c = s.centre_freq
            #cold data
            h5.select(ants=a.name,pol=pol,channels=~static_flags, targets = 'OFF',scans='track')
            freq = h5.channel_freqs
            if not(Ku): nd_temp = nd.temperature(freq / 1e6)
            cold_data = h5.vis[:].real
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            buff = 1
            n_off = ~(np.roll(on,buff) | np.roll(on,-buff))
            n_on = np.roll(on,buff) & np.roll(on,-buff)
            cold_off = n_off
            cold_on = n_on
            #hot data
            h5.select(ants=a.name,pol=pol,channels=~static_flags,targets = 'Moon',scans='track')
            hot_data = h5.vis[:].real
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            buff = 1
            n_off = ~(np.roll(on,buff) | np.roll(on,-buff))
            n_on = np.roll(on,buff) & np.roll(on,-buff)
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
                plt.figure(1)
                plt.subplot(n_ants,2,p)
                plt.ylim(10,25)
                plt.ylabel('T_ND [K]')
                plt.xlim(900,1670)
                plt.xlabel('f [MHz]')
                if p ==ant_num * 2-1: plt.ylabel(ant)
                plt.plot(freq/1e6,Tdiode,'b.',label='Measurement: Y-method')
                #outfile = file('%s/%s.%s.%s.csv' % (output_dir,ant, diode, pol.lower()), 'w')
                #outfile.write('#\n# Frequency [Hz], Temperature [K]\n')
                # Write CSV part of file
                #outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(f[((fs>1.2e9) & (fs < 1.95e9))],d[((fs>1.2e9) & (fs < 1.95e9))])]))
                #outfile.close()
                plt.plot(freq/1e6,nd_temp,'k.',label='Model: EMSS')
                plt.grid()
                plt.legend()
                
            plt.figure(2)
            plt.subplot(n_ants,2,p)
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
        plt.figure(1)
        plt.subplot(n_ants,2,1)
        plt.title('Coupler Diode: H pol: '+file_base)
        plt.subplot(n_ants,2,2)
        plt.title('Coupler Diode: V pol: '+file_base)

    plt.figure(2)
    plt.subplot(n_ants,2,1)
    plt.title('Tsys/eta_A: H pol: '+file_base)
    plt.subplot(n_ants,2,2)
    plt.title('Tsys/eta_A: V pol: '+file_base)

    if pdf:
        if not(Ku):
            fig1.savefig(pp,format='pdf')
            plt.close(fig1)
        fig2.savefig(pp,format='pdf')
        plt.close(fig2)
        pp.close() # close the pdf file


# test main method for the library
if __name__ == "__main__":
#test the method with a know file
    filename = '/var/kat/archive/data/RTS/telescope_products/2015/04/22/1429706275.h5'
    pdf=True
    Ku=False
    verbose=False
    out = '.'
    print 'Performing test run with: ' + filename
    read_and_plot_data(filename,out,pdf,Ku,verbose)
