#!/usr/bin/env python
# plotH5data.py , last mod 03/02/2016 [NJY]

#########################################################
# Retrieve time series and bandpass data from hdf5 file.
# Also plot noise diode and receiver models for receiver 
# serial numbers associated with file.
##########################################################

import sys, glob, time, os, katdal
from optparse import OptionParser   # enables variables to be parsed from command line
from katmisc.utils.timestamp import Timestamp
import numpy as np
import matplotlib.pyplot as p
from matplotlib.ticker import AutoMinorLocator

##############################
# Define useful functions.
##############################
def get_autos(h5):
    """Get array indices for auto-correlation data products."""
    indices = []
    for index in np.arange(h5.corr_products.shape[0]):
        corr = h5.corr_products[index]
        if ( corr[0] == corr[1] ):
            indices.append(index)
    indices = np.int_(indices)
    return indices

def get_UTCs(h5):
    """Convert seconds format timestamp to UTC time."""
    start = str(Timestamp(h5.start_time))[:-1]
    end = str(Timestamp(h5.end_time))[:-1]
    return start, end

def get_rx_models(h5,ant,rDir):
    """Get receiver model and serial number for data set."""
    Band,SN = h5.receivers[ant].split('.')
    recModH = str("{}/Rx{}_SN{:0>4d}_calculated_noise_H_chan.dat".format(rDir,str.upper(Band),int(SN)))
    recModV = str("{}/Rx{}_SN{:0>4d}_calculated_noise_V_chan.dat".format(rDir,str.upper(Band),int(SN)))
    return [recModH, recModV]

def get_nd_models(h5,ant,nDir):
    """Get noise-diode model and serial number for data set."""
    Band,SN = h5.receivers[ant].split('.')
    noiseModH = str("{}/rx.{}.{}.h.csv".format(nDir,str.lower(Band),int(SN)))
    noiseModV = str("{}/rx.{}.{}.v.csv".format(nDir,str.lower(Band),int(SN)))
    return [noiseModH, noiseModV]

def plot_data(fig,corrProds,x,y,labels,autoCorr,logSwitch,coords,waterfall):
    """Plot ND models, Rx models, time-series and bandpass data."""
    xmin = 1e9
    xmax = 0
    ax = p.subplot2grid((2,2+Nant),coords,rowspan=1, colspan=1)  # ((Nrow, Ncol),(row,col),kwargs**)
    colors = ['k','b','r','g','m','c','y']*3
    lstyles = np.r_[['-']*7,['--']*7,[':']*7]
    for index in np.arange(corrProds.shape[0]):
        if ( autoCorr != None ):
            Label = str.upper(corrProds[index][0])
        else:
            Label = corrProds[index]
        
        if ( len(x.shape) > 1 ):
            xUse = x[index]
            if ( xUse.min() < xmin ):  
                xmin = xUse.min()
            if ( xUse.max() > xmax ):
                xmax = xUse.max()
        else:
            xUse = x
            xmin = x.min()
            xmax = x.max()
        
        xspan = xmax - xmin
        if ( logSwitch == 1 ):
            p.semilogy(xUse,y[index],ls=lstyles[index],color=colors[index],label=Label)
            if ( waterfall is True ) and ( xspan > 500 ):
                if (  xspan < 500 ):
                    ax.xaxis.set_major_locator(p.MultipleLocator(100))
                elif ( 500 < xspan < 1000):
                    ax.xaxis.set_major_locator(p.MultipleLocator(200))
                elif ( 1000 < xspan < 2000):
                    ax.xaxis.set_major_locator(p.MultipleLocator(500)) 
                elif ( xspan > 2000):
                    ax.xaxis.set_major_locator(p.MultipleLocator(1000)) 
        else:
            p.plot(xUse,y[index],ls=lstyles[index],color=colors[index],label=Label)
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            if waterfall:
                ax.xaxis.set_major_locator(p.MultipleLocator(200))  
               
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    p.legend(numpoints=1,loc='upper right')
    p.ylabel(labels[1])
    p.xlabel(labels[0])
    p.xlim(xmin,xmax)
    return fig

def plot_waterfall(fig,h5,Z,title,coords):
    """Plot dynamic spectra of H+V autocorr power data."""
    ax = p.subplot2grid((2,2+Nant),coords,rowspan=2, colspan=1)  # ((Nrow, Ncol),(row,col),kwargs**)
    tMax = (h5.timestamps - h5.timestamps[0])[-1]
    freqs = [h5.freqs.min()/1e6,h5.freqs.max()/1e6]
    im = p.imshow(Z, origin='lower',cmap = p.cm.jet, interpolation='nearest',
    aspect='auto', extent=[freqs[0],freqs[1], 0,tMax])
    #CB = p.colorbar(im, orientation='horizontal', shrink=0.7, pad=0.08)
    #CB.set_label('Power (A.U.)')
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))    
    ax.xaxis.set_major_locator(p.MultipleLocator(200))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    p.ylabel('Time (s)')
    p.xlabel('Frequency (MHz)')
    return fig

#############################################
# Initialise parameters from command line.
##############################################
parser = OptionParser(usage="%prog <options>", description='Plots the polarisation properties of SP data.')
parser.add_option("-a", "--autoOnly", default=False, action='store_true', help="Only use auto-corr data (default = False).")
parser.add_option("-c", "--chans", default='2100 2800', type='string', help="Frequency channels to use (default = '2100 2800').")
parser.add_option("-d", "--dir", default=None, type='string', help="Directory to look into for data.")
parser.add_option("-f", "--file", default=None, type='string', help="Input file override.")
parser.add_option("-n", "--noiseDir", default='/var/kat/katconfig/user/noise-diode-models/mkat', type='string', help="Noise-diode model directory to look in (default = %default).")
parser.add_option("-o", "--outfile", default=False, action='store_true', help="Write to file (default = False).")
parser.add_option("-p", "--corrProds", default=None, type='string', help="Correlator products to use (default = all).")
parser.add_option("-r", "--recDir", default='/var/kat/katconfig/user/receiver-models/mkat', type='string', help="Receiver-model directory to look in (default = %default).")
parser.add_option("-s", "--scans", default=None, type='string', help="Space-delimited scans to use (default = all.")
parser.add_option("-t", "--trackOnly", default=False, action='store_true', help="Only use source tracks (default = False.")
parser.add_option("-v", "--verbose", default=False, action='store_true', help="Print verbose information (default = False.")
parser.add_option("-w", "--waterfall", default=False, action='store_true', help="Plot dynamic spectra for autocorr data (default = False.")
(opts, args) = parser.parse_args()
t0 = time.time()  # record script start time

#################################
# Get data from hdf5 input file.
#################################
if ( opts.file != None ):
    if ( opts.dir != None ):
        file = opts.dir + opts.file
    else:
        file = opts.file
else:
    if ( opts.dir != None ):
        files = np.array(glob.glob(opts.dir+"/*.h5"))
        if ( files.size > 1 ):
            print '\n List of files to choose from:\n', files
            print '\n Select file to analyse...'
            file = raw_input()
        else:
            print '\n File selected = ', files[0]            
            file = files[0]
    else:
        print '\n Specify file to analyse. Exiting now...\n'
        sys.exit(1)

h5 = katdal.open(file)
tdump = h5.dump_period
bw = h5.channel_width
freqs = h5.freqs
fShape = h5.shape
descr = '\n'.join(h5.__str__().split('\n')[:23]) # human-friendly header information
sensors = h5.sensor.keys()
if opts.verbose:
    print '\n', descr # print header
else:
    print ''

##################################
# Get model data from csv files.
##################################
aIndex = 0
ants = []
for ant in h5.ants:
    ants.append(str.upper(ant.name))
    rxFiles = get_rx_models(h5,ant.name,opts.recDir)
    noiseFiles = get_nd_models(h5,ant.name,opts.noiseDir)
    for index in np.arange(len(rxFiles)):
        rxFreqs, rxTsys = np.loadtxt(rxFiles[index],usecols=[0,2],skiprows=1,unpack=True,delimiter=',')
        noiseFreqs, noiseTsys = np.loadtxt(noiseFiles[index],usecols=[0,1],skiprows=2,unpack=True,delimiter=',')
        if ( aIndex == 0 ):
            all_rxFreqs = np.copy(rxFreqs)
            all_rxTsys = np.copy(rxTsys)
            all_noiseFreqs = np.copy(noiseFreqs)
            all_noiseTsys = np.copy(noiseTsys)
        else:
            all_rxFreqs = np.vstack((all_rxFreqs,rxFreqs))
            all_rxTsys = np.vstack((all_rxTsys,rxTsys))
            all_noiseFreqs = np.vstack((all_noiseFreqs,noiseFreqs))
            all_noiseTsys = np.vstack((all_noiseTsys,noiseTsys))
        aIndex += 1

global Nant
Nant = len(ants)
ants = ', '.join(ants)

################################
# Select data of interest.
################################
if opts.verbose:
    print '\n Correlator products in file:\n', h5.corr_products
if ( opts.corrProds == None ):
    if ( opts.autoOnly == False ):
        print '\n Enter correlator product array indices to use...' 
        indices = np.int_(raw_input('').split())
    else:
        print '\n Selecting auto-correlation data only...'
        #h5.select(corrprods=auto)
        indices = get_autos(h5)
else:
    print '\n Selecting user-specified correlation products...'
    indices = np.int_(opts.corrProds.split())
h5.select(corrprods=indices)
print '\n Correlator products selected:\n', h5.corr_products

# select scan indices
if ( opts.scans != None ):
    h5.select(scans=np.int_(opts.scans.split))
# select source tracks only
if opts.trackOnly:
    h5.select(scans='track')
tvals = h5.timestamps - h5.timestamps[0] # get updated timestamps (s)

###############################
# Get BP and time-series data.
###############################
print '\n Computing time-series and average bandpasses...'
if ( opts.chans != None ):
    chans = np.int_(opts.chans.split())
else:
    chans = np.arange(h5.channels.shape[0])

allTS = []
allBPs = []
Nprod = h5.corr_products.shape[0]
for index in np.arange(Nprod):
    bp = np.abs(h5.vis[:,:,index]).mean(axis=0)
    ts = np.abs(h5.vis[:,chans[0]:chans[1]+1,index]).mean(axis=1)
    allBPs.append(bp)
    allTS.append(ts)

##########################
# Plot data in subplots.
##########################
print ' Plotting time series and average bandpasses..'
fig = p.figure(num=1, figsize=(13, 8.9), dpi=80)
p.clf()
if (opts.waterfall):
    params = {'axes.labelsize': 14, 'font.size': 15, 'legend.fontsize': 9, 
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'text.usetex': True}
else:
    params = {'axes.labelsize': 18, 'font.size': 15, 'legend.fontsize': 10, 
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'text.usetex': True}
p.rcParams.update(params)

#--------------------------------#
# Plot all TS, BP and Tsys plots:
#--------------------------------#
freqs = freqs/1e6
fig = plot_data(fig,h5.corr_products,tvals,allTS,
    ['Time (s)','Power (A.U.)'],opts.autoOnly,1,(0,0),opts.waterfall)
fig = plot_data(fig,h5.corr_products,freqs,allBPs,
    ['Frequency (MHz)','Power (A.U.)'],opts.autoOnly,1,(0,1),opts.waterfall)
fig = plot_data(fig,h5.corr_products,all_rxFreqs/1e6,all_rxTsys,
    ['Frequency (MHz)',r'$T_{\mathrm{rx}}$ (K)'],opts.autoOnly,0,(1,0),opts.waterfall)
fig = plot_data(fig,h5.corr_products,all_noiseFreqs/1e6,all_noiseTsys,
    ['Frequency (MHz)',r'$T_{\mathrm{nd}}$ (K)'],opts.autoOnly,0,(1,1),opts.waterfall)

#-------------------------------#
# Plot autocorr dynamic spectra:
#-------------------------------#
Nchan = h5.freqs.shape[0]
if opts.waterfall:
    print ' Plotting autocorr dynamic spectra...'
    if ( opts.autoOnly is False ):
        indices = get_autos(h5)
        h5.select(corrprods=indices)
    index = aIndex = 0
    Nprods = h5.corr_products.shape[0]
    while ( index < Nprods ):
        ant = str.upper(h5.ants[aIndex].name)
        tfCube = np.abs(h5.vis[:,:,index]).reshape((-1,Nchan)) + np.abs(h5.vis[:,:,index+1]).reshape((-1,Nchan))
        fig = plot_waterfall(fig,h5,tfCube,ant,(0,2+aIndex))
        p.title(ant,fontsize=10)
        index += 2
        aIndex += 1

#----------------------------------------#
# Write figure title and optionally save:
#----------------------------------------#
times = get_UTCs(h5)
suptext = 'filename = '+ file.split('/')[-1] + '; ' + r'$\mathrm{UTC}_{\mathrm{start}}$'\
 + ' = ' + times[0] + '; ants = ' + ants
p.suptitle(suptext, fontsize=10, fontweight='normal',y=0.99)
p.subplots_adjust(left=0.075,right=0.97,top=0.95,bottom=0.075,hspace=0.2,wspace=0.2)
if opts.waterfall: 
    p.subplots_adjust(hspace=0.2,wspace=0.3)    
if opts.outfile:
    outfile = '/home/kat/'+file.split('/')[-1].split('.h5')[0]+'_miscPlots.pdf'
    p.savefig(outfile,format='pdf')
t1 = time.time() # record script finish time
p.show()

#*******************************#
print '\n Total elapsed time: %.2f s (%.2f mins)\n' %(t1-t0,(t1-t0)/60.)
