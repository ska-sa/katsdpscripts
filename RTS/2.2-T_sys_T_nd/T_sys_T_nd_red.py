# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import katarchive
import katdal as katfile
import scape

import numpy as np

import optparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pols = ['v','h']
diode = 'coupler' 


parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description=" This produces a pdf file with graphs verifying the ND model and Tsys for each antenna in the file")
parser.add_option("-f", "--frequency-bandwidth", dest="freq_band", type="float", default='256e6',
                  help="BAndwidth of frequency channels to keep. Default = %default")

(opts, args) = parser.parse_args()

if len(args) ==0:
    raise RuntimeError('Please specify an h5 file to load.')


###
### Make a nice file name
### And open a pdf file for writing
###
filename = args[0]
nice_filename =  filename.split('/')[-1]+ '_T_sys_T_nd'
pp = PdfPages(nice_filename+'.pdf')

h5 = katfile.open(filename)

ants = h5.ants

colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
fig1 = plt.figure(1,figsize = (10,16))
fig2 = plt.figure(2,figsize = (10,16))

for pol in pols:
    for a,col in zip(ants,colour):    
        ant = a.name
        ant_num = int(ant[3])
        if ant_num>7:
            continue
        air_temp = np.mean(h5.sensor['Enviro/asc.air.temperature'])
        nd_model = h5.file['MetaData/Configuration/Antennas/'+ant+'/'+pol+'_'+diode+'_noise_diode_model'].value
        nd = scape.gaincal.NoiseDiodeModel(freq = nd_model[:,0]/1e6,temp = nd_model[:,1])
        
        for s_i,s in enumerate(h5.spectral_windows):
            if s_i == 0 and s.centre_freq != 1264e6:
                continue
            f_c = s.centre_freq
            #cold data
            h5.select(ants=a.name,spw=s_i,pol=pol,freqrange=(f_c - opts.freq_band/2, f_c + opts.freq_band/2),targets = 'OFF',scans='track')
            freq = h5.channel_freqs
            nd_temp = nd.temperature(freq / 1e6)
            cold_data = np.ma.array(h5.vis[:].real,mask=h5.flags()[:],fill_value=np.nan)
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            buff = 5
            n_off = ~(np.roll(on,buff) | np.roll(on,-buff))
            n_on = np.roll(on,buff) & np.roll(on,-buff)
            cold_off = n_off
            cold_on = n_on
            #hot data
            h5.select(ants=a.name,spw=s_i,pol=pol,freqrange=(f_c - opts.freq_band/2, f_c + opts.freq_band/2),targets = 'Moon',scans='track')
            hot_data = np.ma.array(h5.vis[:].real,mask=h5.flags()[:],fill_value=np.nan)
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            buff = 5
            n_off = ~(np.roll(on,buff) | np.roll(on,-buff))
            n_on = np.roll(on,buff) & np.roll(on,-buff)
            hot_off = n_off
            hot_on = n_on
            cold_spec = np.median(cold_data[cold_off,:,0].filled(),0)
            hot_spec = np.median(hot_data[hot_off,:,0].filled(),0)
            cold_nd_spec = np.median(cold_data[cold_on,:,0].filled(),0)
            hot_nd_spec = np.median(hot_data[hot_on,:,0].filled(),0)
            continue

        
        
        hs,f = hot_spec,freq
        cs = cold_spec
        hns = hot_nd_spec
        cns = cold_nd_spec
        Y = hs / cs
        HPBW = 1.22 * (180/np.pi) *(3e8/(12*f))
        om = 1.133 * HPBW**2
        R = 0.25
        Thot = 225 * (np.pi * R**2)/om 
        Tsys = (Thot)/(Y-1)
        Ydiode = hns / cns
        Tdiode = (Thot + Tsys*(1-Ydiode))/(Ydiode-1)
        

        
        plt.figure(1)
        if pol == 'v' : p = ant_num * 2 
        if pol == 'h' : p = ant_num * 2-1 
        plt.subplot(7,2,p)
        plt.ylim(1.5 ,6)
        if p ==ant_num * 2-1: plt.ylabel(ant)
        plt.plot(f,Tdiode,'b')
        #outfile = file('%s.%s.%s.csv' % (ant, diode, pol.lower()), 'w')
        #outfile.write('#\n# Frequency [Hz], Temperature [K]\n')
        # Write CSV part of file
        #outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(f[((fs>1.2e9) & (fs < 1.95e9))],d[((fs>1.2e9) & (fs < 1.95e9))])]))
        #outfile.close()
        plt.plot(f,nd_temp,'k')
        plt.figure(2)
        p = ant_num * 2 if pol == 'v' else ant_num * 2 -1
        plt.subplot(7,2,p)
        plt.ylim(10,60)
        if p == ant_num * 2 -1: plt.ylabel(ant)
        plt.plot(f,Tsys,'b')
        plt.grid()
    
plt.figure(1)
plt.subplot(7,2,1)
plt.title('Coupler Diode: H pol')
plt.subplot(7,2,2)
plt.title('Coupler Diode: V pol')

plt.figure(2)
plt.subplot(7,2,1)
plt.title('Tsys: H pol')
plt.subplot(7,2,2)
plt.title('Tsys: V pol')

fig1.savefig(pp,format='pdf')
plt.close(fig1)
fig2.savefig(pp,format='pdf')
plt.close(fig2)
pp.close() # close the pdf file



