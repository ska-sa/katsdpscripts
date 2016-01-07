#!/usr/bin/python
#Plots uncalibrated power, noise diode firings, derived gains to assess gain stability and effectiveness of the gain calibration
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt   
import optparse
import katdal
from matplotlib.backends.backend_pdf import PdfPages
import stefcal
import pandas
import os
from katsdpscripts.RTS import git_info
from astropy.time import Time
import pickle

def polyfitstd(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Modified polyfit:Any nan values in x are 'masked' and std is returned .
    """
    if np.isnan(y).sum() >= y.shape[0]+1 : return 0.0
    z = np.ma.array(data=np.nan_to_num(y),mask=np.isnan(y))
    if x.shape[0] <= deg +1  :
        z = np.zeros((deg+1))
        z[-1] = np.ma.mean(x)
        return z[0]
    gg = np.ma.polyfit(x, z, deg, rcond, full, w, cov)
    #if np.isnan(gg[0]) : raise RuntimeError('NaN in polyfit, Error')
    return anglestd(z-x*gg[0])

def detrend(x):
    return polyfitstd(np.arange(x.shape[0]),x,1)

def cplot(data,*args,**kwargs):
    if data.dtype.kind == 'c': plot(np.real(data),np.imag(data),*args,**kwargs)
    else : plot(data,*args,**kwargs)

def mean(a,axis=None):
    """This function calclates the mean along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the mean of the argument & the angle (exp(1j*theta) )
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    >>>>mean(np.array((1+1j , 1-1j)))
    (1.4142135623730951+0j)
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real**2 + a.imag**2).mean(axis=axis)
        th = np.arctan2(a.imag,a.real)
        sa = (np.sin(th)).sum(axis=axis)
        ca = (np.cos(th)).sum(axis=axis)
        thme = np.arctan2(sa,ca)
        return r*np.exp(1j*thme)
    else:
        return np.mean(a,axis=axis)


def yamartino_method(a,axis=None):
    """This function calclates the standard devation along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the standard devation of the argument & the angle (exp(1j*theta) )
    This uses the Yamartino method which is a one pass method of estimating the standard devation 
    of an angle
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real**2 + a.imag**2).std(axis=axis)#mean
        th = np.arctan2(a.imag,a.real)
        if axis is None :
            sa = (np.sin(th)/len(th)).sum()
            ca = (np.cos(th)/len(th)).sum()
        else:
            sa = (np.sin(th)/len(th)).sum(axis=axis)
            ca = (np.cos(th)/len(th)).sum(axis=axis)
        e = np.sqrt(1.-(sa**2+ca**2))
        thsd = np.arcsin(e)*(1.+(2./np.sqrt(3)-1.)*e**3)
        return r*np.exp(1j*thsd)
    else:
        return np.std(a,axis=axis)

def std(a,axis=None):
    """This function calclates the standard devation along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the standard devation of the argument & the angle (exp(1j*theta) )
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
    The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real**2 + a.imag**2).std(axis=axis)#mean
        th = np.arctan2(a.imag,a.real)
        sa = np.sin(th).sum(axis=axis)
        ca = np.cos(th).sum(axis=axis)
        thme = np.arctan2(sa,ca)        
        nshape = np.array(th.shape)
        nshape[axis] = 1
        S0 = 1-np.cos(th-np.reshape(thme,nshape)).mean(axis=axis)
        return r*np.exp(1j*np.sqrt(-2.*np.log(1.-S0)))
    else:
        return np.std(a,axis=axis)


def absstd(a,axis=None):
    """This function calclates the standard devation along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the standard devation  of the angle (exp(1j*theta) )
    and standard devation over the mean of the argument
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    a = np.ma.array(data=np.nan_to_num(a),mask=np.isnan(a))
    if a.dtype.kind == 'c':
        rstd = np.ma.sqrt(a.real**2 + a.imag**2).std(axis=axis)#std
        rmean = np.ma.sqrt(a.real**2 + a.imag**2).mean(axis=axis)
        th = np.ma.arctan2(a.imag,a.real)
        sa = np.ma.sin(th).sum(axis=axis)
        ca = np.ma.cos(th).sum(axis=axis)
        thme = np.ma.arctan2(sa,ca)        
        nshape = np.array(th.shape)
        nshape[axis] = 1
        S0 = 1-np.ma.cos(th-np.ma.reshape(thme,nshape)).mean(axis=axis)
        return (rstd/rmean)*np.exp(1j*np.ma.sqrt(-2.*np.log(1.-S0)))
    else:
        return np.std(a,axis=axis)/np.mean(a,axis=axis)

def anglestd(a,axis=None):
    """This function calclates the standard devation along the chosen axis of the array
    This function has been writen to calculate the mean of complex numbers correctly
    by taking the standard devation  of the angle (exp(1j*theta) )
    and standard devation over the mean of the argument
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    a = np.ma.array(data=np.nan_to_num(np.exp(1j* a)),mask=np.isnan(a))
    rstd = np.ma.sqrt(a.real**2 + a.imag**2).std(axis=axis)#std
    rmean = np.ma.sqrt(a.real**2 + a.imag**2).mean(axis=axis)
    th = np.ma.arctan2(a.imag,a.real)
    sa = np.ma.sin(th).sum(axis=axis)
    ca = np.ma.cos(th).sum(axis=axis)
    thme = np.ma.arctan2(sa,ca)        
    nshape = np.array(th.shape)
    nshape[axis] = 1
    S0 = 1-np.ma.cos(th-np.ma.reshape(thme,nshape)).mean(axis=axis)
    return np.degrees(np.angle(np.exp(1j*np.ma.sqrt(-2.*np.log(1.-S0)))))


def plot_AntennaGain(gains,freq,inputs):
    """ This plots the Amplitude and Phase across the bandpass
    gains   : complex array shape (time,frequency,antenna) or (frequency,antenna)  
    freq    : array shape (frequency) in Herts
    inputs  : list of string names of the antennas
    returns : a list of figure objects
    """
    returned_plots = []
    if len(gains.shape) == 2 : gains =gains[np.newaxis,:,:]
    for i in xrange(gains.shape[-1]):
        fig, (ax) = plt.subplots(nrows=2, sharex=True)
        ax[0].set_title('Phase %s'%(inputs[i]))
        ax[1].set_title('Amplitude %s'%(inputs[i]))
        ax[0].set_ylabel('Degrees')
        ax[1].set_ylim(0,20)
        ax[1].set_xlabel("Frequency (MHz)")
        ax[0].plot(freq/1e6,np.degrees(np.angle(gains[:,:,i].T)))
        ax[1].plot(freq/1e6,np.abs(gains[:,:,i].T))
        returned_plots.append(fig)
    return returned_plots



def  fringe_stopping(data): #This will have to be updated for MKAT
    new_ants = {
    'ant1' : ('25.0950 -9.0950 0.0450', 23220.506e-9, 23228.551e-9),
    'ant2' : ('90.2844 26.3804 -0.22636', 23283.799e-9, 23286.823e-9),
    'ant3' : ('3.98474 26.8929 0.0004046', 23407.970e-9, 23400.221e-9),
    'ant4' : ('-21.6053 25.4936 0.018615', 23514.801e-9, 23514.801e-9),
    'ant5' : ('-38.2720 -2.5917 0.391362', 23676.033e-9, 23668.223e-9),
    'ant6' : ('-61.5945 -79.6989 0.701598', 23782.854e-9, 23782.150e-9),
    'ant7' : ('-87.9881 75.7543 0.138305', 24047.672e-9, 24039.237e-9),}
    delays = {}
    for inp in data.inputs:
        ant, pol = inp[:-1], inp[-1]
        delays[inp] = new_ants[ant][1 if pol == 'h' else 2]
    center_freqs = data.channel_freqs
    wavelengths = 3.0e8 / center_freqs
    # Number of turns of phase that signal B is behind signal A due to cable / receiver delay
    cable_delay_turns = np.array([(delays[inpB] - delays[inpA]) * center_freqs for inpA, inpB in data.corr_products]).T
    crosscorr = [(data.inputs.index(inpA), data.inputs.index(inpB)) for inpA, inpB in data.corr_products]
    # Assemble fringe-stopped visibility data for main (bandpass) calibrator
    vis_set = None
    for compscan_no,compscan_label,target in data.compscans():
        print "loop",compscan_no,compscan_label,target
        vis = data.vis[:,:,:]
        # Number of turns of phase that signal B is behind signal A due to geometric delay
        geom_delay_turns = - data.w[:, np.newaxis, :] / wavelengths[:, np.newaxis]
        # Visibility <A, B*> has phase (A - B), therefore add (B - A) phase to stop fringes (i.e. do delay tracking)
        vis *= np.exp(2j * np.pi * (geom_delay_turns + cable_delay_turns))
        if vis_set is None:
            vis_set = vis.copy()
        else:
            vis_set = np.append(vis_set,vis,axis = 0)
    return vis_set


def peak2peak(y):
    return np.degrees(np.ma.ptp(np.ma.angle(np.ma.array(data=np.nan_to_num(y),mask=np.isnan(y)))))

def calc_stats(timestamps,gain,pol='no polarizarion',windowtime=1200,minsamples=1):
    """ calculate the Stats needed to evaluate the observation"""
    returntext = []
    #note gain is in radians
    gain_ts = pandas.Series(gain, pandas.to_datetime(np.round(timestamps), unit='s'))
    window_occ = pandas.rolling_count(gain_ts,windowtime)/float(windowtime)
    full = np.where(window_occ==1)
    #note std is returned in degrees
    std = (pandas.rolling_apply(gain_ts,windowtime,anglestd,minsamples)).iloc[full]    
    peakmin= (np.degrees(pandas.rolling_min(gain_ts,windowtime,minsamples))).iloc[full]
    peakmax= (np.degrees(pandas.rolling_max(gain_ts,windowtime,minsamples))).iloc[full]
    peak = peakmax-peakmin
    dtrend_std = (pandas.rolling_apply(gain_ts,windowtime,detrend,minsamples)).iloc[full]
    #trend_std = pandas.rolling_apply(ts,5,lambda x : np.ma.std(x-(np.arange(x.shape[0])*np.ma.polyfit(np.arange(x.shape[0]),x,1)[0])),1)
    timeval = timestamps.max()-timestamps.min()
    
    
    #rms = np.sqrt((gain**2).mean())
    returntext.append("Total time of observation : %f (seconds) with %i accumulations."%(timeval,timestamps.shape[0]))
    #returntext.append("The mean gain of %s is: %.5f"%(pol,gain.mean()))
    #returntext.append("The Std. dev of the gain of %s is: %.5f"%(pol,gain.std()))
    #returntext.append("The RMS of the gain of %s is : %.5f"%(pol,rms))
    #returntext.append("The Percentage variation of %s is: %.5f"%(pol,gain.std()/gain.mean()*100))
    returntext.append("The mean Peak to Peak range over %i seconds of %s is: %.5f (req < 13 )  "%(windowtime,pol,peak.mean()))
    returntext.append("The Max Peak to Peak range over %i seconds of %s is: %.5f  (req < 13 )  "%(windowtime,pol,peak.max()))
    returntext.append("The mean variation over %i seconds of %s is: %.5f    "%(windowtime,pol,std.mean()))
    returntext.append("The Max  variation over %i seconds of %s is: %.5f    "%(windowtime,pol,std.max()))
    returntext.append("The mean detrended variation over %i seconds of %s is: %.5f    (req < 2.3 )"%(windowtime,pol,dtrend_std.mean()))
    returntext.append("The Max  detrended variation over %i seconds of %s is: %.5f    (req < 2.3 )"%(windowtime,pol,dtrend_std.max()))
    #a - np.round(np.polyfit(b,a.T,1)[0,:,np.newaxis]*b + np.polyfit(b,a.T,1)[1,:,np.newaxis])
    
    pltobj = plt.figure(figsize=[8,11])
    plt.suptitle(h5.name)
    plt.subplots_adjust(bottom=0.15, hspace=0.35, top=0.95)
    ax1 = plt.subplot(311)
    plt.title('Original unwrapped phases for '+pol)
    np.degrees(gain_ts).plot(label='gain phase')
    peakmax.plot(label='rolling max')
    peakmin.plot(label='rolling min')
    plt.legend(loc='best')
    plt.ylabel('Gain phase (deg)') 
    ax2 = plt.subplot(312)
    plt.title('Peak to peak variation of %s, %i Second sliding Window'%(pol,windowtime,))
    peak.plot(color='blue')
    ax2.axhline(13,ls='--', color='red')
    plt.ylabel('Variation (deg)')

    ax3  = plt.subplot(313)
    plt.title('Variation of %s, %i Second sliding Window'%(pol,windowtime,))
    dtrend_std.plot(label='Detrended std')
    ax3.axhline(2.3,ls='--', color='red')
    plt.ylabel('Variation (deg)')
    plt.xlabel('Date/Time')
    plt.legend(loc='best')
    plt.figtext(0.89, 0.05, git_info(), horizontalalignment='right',fontsize=10)
    return returntext,pltobj  # a plot would be cool



# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description=" This produces a pdf file with graphs decribing the gain stability for each antenna in the file")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", type="string", default='211,3896',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")
parser.add_option("-o","--output_dir", default='.', help="Output directory for pdfs. Default is cwd")
parser.add_option("-c", "--channel-mask", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle', 
                  help="Optional pickle file with boolean array specifying channels to mask (Default = %default)")
(opts, args) = parser.parse_args()

if len(args) ==0:
    raise RuntimeError('Please specify an h5 file to load.')
    


# frequency channels to keep
start_freq_channel = int(opts.freq_keep.split(',')[0])
end_freq_channel = int(opts.freq_keep.split(',')[1])

h5 = katdal.open(args)
n_chan = np.shape(h5.channels)[0]
if not opts.freq_keep is None :
    start_freq_channel = int(opts.freq_keep.split(',')[0])
    end_freq_channel = int(opts.freq_keep.split(',')[1])
    edge = np.tile(True, n_chan)
    edge[slice(start_freq_channel, end_freq_channel)] = False
else :
    edge = np.tile(False, n_chan)
#load static flags if pickle file is given
if len(opts.channel_mask)>0:
    pickle_file = open(opts.channel_mask)
    rfi_static_flags = pickle.load(pickle_file)
    pickle_file.close()
else:
    rfi_static_flags = np.tile(False, n_chan)

static_flags = np.logical_or(edge,rfi_static_flags)


fileprefix = os.path.join(opts.output_dir,os.path.splitext(args[0].split('/')[-1])[0])
nice_filename =  fileprefix+ '_phase_stability'
pp = PdfPages(nice_filename+'.pdf')
for pol in ('h','v'):
    h5.select(channels=~static_flags,pol=pol,corrprods='cross',scans='track',dumps=slice(1,600)) 
    # loop over both polarisations
    if np.all(h5.sensor['CorrelatorBeamformer/auto_delay_enabled'] == '0') :
        print "Need to do fringe stopping "
        vis = fringe_stopping(h5)
    else:
        print "Fringe stopping done in the correlator"
        vis = h5.vis[:,:,:]

    flaglist = ~h5.flags()[:,:,:].any(axis=0).any(axis=-1)
    #flaglist[0:start_freq_channel] = False
    #flaglist[end_freq_channel:] = False
    antA = [h5.inputs.index(inpA) for inpA, inpB in h5.corr_products]
    antB = [h5.inputs.index(inpB) for inpA, inpB in h5.corr_products]

    N_ants = len(h5.ants)
    #full_vis = np.concatenate((vis, vis.conj()), axis=-1)
    full_antA = np.r_[antA, antB]
    full_antB = np.r_[antB, antA]

    weights= np.abs(1./np.angle(absstd(vis[:,:,:],axis=0)))
    weights= np.concatenate((weights, weights), axis=-1)

    # use vector mean == np.mean  on visabilitys
    # but use angle mean on solutions/phase change.
    gains = stefcal.stefcal( np.concatenate( (np.mean(vis,axis=0), np.mean(vis.conj(),axis=0)), axis=-1) , N_ants, full_antA, full_antB, num_iters=50,weights=weights)
    calfac = 1./(gains[np.newaxis][:,:,full_antA]*gains[np.newaxis][:,:,full_antB].conj())

    h5.select(channels=~static_flags,pol=pol,corrprods='cross',scans='track')
    data = np.zeros((h5.shape[0:3:2]),dtype=np.complex)
    i = 0
    for scan in h5.scans():
        print scan
        if np.all(h5.sensor['CorrelatorBeamformer/auto_delay_enabled'] == '0') :
            print "stopping fringes for size ",h5.shape
            vis = fringe_stopping(h5)
        else:
            vis = h5.vis[:,:,:]
        data[i:i+h5.shape[0]] = mean((vis*calfac[:,:,:h5.shape[-1]])[:,flaglist,:],axis=1)
        i += h5.shape[0]
    figlist = []
    figlist += plot_AntennaGain(gains,h5.channel_freqs,h5.inputs)
    fig = plt.figure()
    plt.suptitle(h5.name)
    plt.title('Phase angle in Baseline vs. Time for %s pol baselines '%(pol))
    plt.imshow(np.degrees(np.angle(data)),aspect='auto',interpolation='nearest')
    #ax = plt.subplot(111)
    #ax.yaxis_date()
    plt.ylabel('Time, (colour angle in degrees)');plt.xlabel('Baseline Number')
    plt.colorbar()
    fig.savefig(pp,format='pdf')
    plt.close(fig)
    
    for i,(ant1,ant2) in  enumerate(h5.corr_products):
        print "Generating Stats on the baseline %s,%s"%(ant1,ant2)
        returntext,pltfig = calc_stats(h5.timestamps[:],np.unwrap(np.angle(data[:,i])) ,pol="%s,%s"%(ant1,ant2),windowtime=1200,minsamples=1)
        pltfig.savefig(pp,format='pdf') 
        plt.close(pltfig)
        fig = plt.figure(None,figsize = (10,10))
        plt.figtext(0.1,0.5,'\n'.join(returntext),fontsize=10)
        fig.savefig(pp,format='pdf')
        plt.close(fig)

#figlist += plot_DataStd(gains,freq,h5.inputs)
#figlist += plot_DataStd(vis,freq,h5.corr_products)
#returntext = calc_stats(d.timestamps,g_hh,d.freqs,'HH',1200)+calc_stats(d.timestamps,g_vv,d.freqs,'VV',1200)
#fig = plt.figure(None,figsize = (10,16))
#plt.figtext(0.1,0.1,'\n'.join(returntext),fontsize=10)
#fig.savefig(pp,format='pdf')
pp.close()
plt.close('all')


#data = mean(rolling_window(mean(vis[:,flaglist,:]*calfac[:,flaglist,:calfac.shape[-1]//2],axis=1),50,axis=0),axis=-1)



#for i in xrange(h5.shape[0]) :plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[i,200:800,0][0,:,0]),'b.',alpha=0.1)
#plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[:,200:800,0].mean(axis=0)),'g',)
#plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[:,200:800,0].mean(axis=0))+3*np.abs(h5.vis[:,200:800,0].std(axis=0)),'r')
#plot(h5.channel_freqs[200:800]/1e6,np.abs(h5.vis[:,200:800,0].mean(axis=0))-3*np.abs(h5.vis[:,200:800,0].std(axis=0)),'r')



