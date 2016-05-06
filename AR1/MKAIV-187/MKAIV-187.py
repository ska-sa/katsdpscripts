#!/usr/bin/python
#Plots uncalibrated power, noise diode firings, derived gains to assess gain stability and effectiveness of the gain calibration
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt   
import optparse
import katdal
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas

def stefcal(vis, num_ants, antA, antB, weights=1.0, num_iters=10, ref_ant=0, init_gain=None):
    """Solve for antenna gains using StefCal (array dot product version).

    The observed visibilities are provided in a NumPy array of any shape and
    dimension, as long as the last dimension represents baselines. The gains
    are then solved in parallel for the rest of the dimensions. For example,
    if the *vis* array has shape (T, F, B) containing *T* dumps / timestamps,
    *F* frequency channels and *B* baselines, the resulting gain array will be
    of shape (T, F, num_ants), where *num_ants* is the number of antennas.

    In order to get a proper solution it is important to include the conjugate
    visibilities as well by reversing antenna pairs, e.g. by forming

    full_vis = np.concatenate((vis, vis.conj()), axis=-1)
    full_antA = np.r_[antA, antB]
    full_antB = np.r_[antB, antA]

    Parameters
    ----------
    vis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    antA, antB : array of int, shape (N,)
        Antenna indices associated with visibilities
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    num_iters : int, optional
        Number of iterations
    ref_ant : int, optional
        Index of reference antenna that will be forced to have a gain of 1.0
    init_gain : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)

    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna

    Notes
    -----
    The model visibilities are assumed to be 1, implying a point source model.

    The algorithm is iterative but should converge in a small number of
    iterations (10 to 30).

    """
    # Each row of this array contains the indices of baselines with the same antA
    baselines_per_antA = np.array([(antA == m).nonzero()[0] for m in range(num_ants)])
    # Each row of this array contains corresponding antB indices with same antA
    antB_per_antA = antB[baselines_per_antA]
    weighted_vis = weights * vis
    weighted_vis = weighted_vis[..., baselines_per_antA]
    # Initial estimate of gain vector
    gain_shape = tuple(list(vis.shape[:-1]) + [num_ants])
    g_curr = np.ones(gain_shape, dtype=np.complex) if init_gain is None else init_gain
    for n in range(num_iters):
        # Basis vector (collection) represents gain_B* times model (assumed 1)
        g_basis = g_curr[..., antB_per_antA]
        # Do scalar least-squares fit of basis vector to vis vector for whole collection in parallel
        g_new = (g_basis * weighted_vis).sum(axis=-1) / (g_basis.conj() * g_basis).sum(axis=-1)
        # Normalise g_new to match g_curr so that taking their average and diff
        # make sense (without copy() the elements of g_new are mangled up)
        g_new /= g_new[..., ref_ant][..., np.newaxis].copy()
        print "Iteration %d: mean absolute gain change = %f" % \
              (n + 1, 0.5 * np.abs(g_new - g_curr).mean())
        # Avoid getting stuck during iteration
        g_curr = 0.5 * (g_new + g_curr)
    return g_curr





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





def  fringe_stopping(data):
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
    """ calculate the Stats needed to evaluate the obsevation"""
    returntext = []
    gain_ts = pandas.Series(gain, pandas.to_datetime(np.round(timestamps), unit='s')).asfreq(freq='1s')
    #mean = pandas.rolling_mean(gain_ts,windowtime,minsamples)
    std = pandas.rolling_apply(gain_ts,windowtime,anglestd,minsamples)    
    peakmin= pandas.rolling_min(gain_ts,windowtime,minsamples)
    peakmax= pandas.rolling_max(gain_ts,windowtime,minsamples)
    peak = peakmax-peakmin
    dtrend_std = pandas.rolling_apply(gain_ts,windowtime,detrend,minsamples)
    #trend_std = pandas.rolling_apply(ts,5,lambda x : np.ma.std(x-(np.arange(x.shape[0])*np.ma.polyfit(np.arange(x.shape[0]),x,1)[0])),1)
    timeval = timestamps.max()-timestamps.min()
    #window_occ = pandas.rolling_count(gain_ts,windowtime)/float(windowtime)
    
    #rms = np.sqrt((gain**2).mean())
    returntext.append("Total time of obsevation : %f (seconds) with %i accumulations."%(timeval,timestamps.shape[0]))
    #returntext.append("The mean gain of %s is: %.5f"%(pol,gain.mean()))
    #returntext.append("The Std. dev of the gain of %s is: %.5f"%(pol,gain.std()))
    #returntext.append("The RMS of the gain of %s is : %.5f"%(pol,rms))
    #returntext.append("The Percentage variation of %s is: %.5f"%(pol,gain.std()/gain.mean()*100))
    returntext.append("The mean Peak to Peak range over %i seconds of %s is: %.5f  (req < 3 )"%(windowtime,pol,np.degrees(peak.mean())))
    returntext.append("The Max Peak to Peak range over %i seconds of %s is: %.5f   (req < 3 )"%(windowtime,pol,np.degrees(peak.max())))
    returntext.append("The mean variation over %i seconds of %s is: %.5f    (req < 2.5 )"%(windowtime,pol,np.degrees(std.mean())))
    returntext.append("The Max  variation over %i seconds of %s is: %.5f    (req < 2.5 )"%(windowtime,pol,np.degrees(std.max())))
    returntext.append("The mean detrended variation over %i seconds of %s is: %.5f    (req < 2.3 )"%(windowtime,pol,np.degrees(dtrend_std.mean())))
    returntext.append("The Max  detrended variation over %i seconds of %s is: %.5f    (req < 2.3 )"%(windowtime,pol,np.degrees(dtrend_std.max())))
    #a - np.round(np.polyfit(b,a.T,1)[0,:,np.newaxis]*b + np.polyfit(b,a.T,1)[1,:,np.newaxis])
    
    pltobj = plt.figure()
    plt.title('Variation of %s, %i Second sliding Window'%(pol,windowtime,))
    std.plot(label='Orignal')
    dtrend_std.plot(label='Detrended')
    #window_occ.plot(label='Window Occupancy')
    plt.hlines(2.3, timestamps.min(), timestamps.max(), colors='k')
    plt.hlines(2.5, timestamps.min(), timestamps.max(), colors='k')
    plt.hlines(3, timestamps.min(), timestamps.max(), colors='k')
    plt.ylabel('Variation')
    plt.xlabel('Date/Time')
    plt.legend(loc='best')
    #plt.title(" %s pol Gain"%(pol))
    #plt.plot(windowgainchange.mean(),'b',label='20 Min (std/mean)')
    #plt.plot(np.ones_like(windowgainchange.mean())*2.0,'r',label=' 2 level')
    return returntext,pltobj  # a plot would be cool



# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description=" This produces a pdf file with graphs decribing the gain sability for each antenna in the file")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", type="string", default='2000,3000',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")

parser.add_option("-o","--output_dir", default='.', help="Output directory for pdfs. Default is the cwd")
parser.add_option("-c", "--channel-mask", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle',
                  help="Optional pickle file with boolean array specifying channels to mask (Default = %default)")
parser.add_option("-r", "--rfi-flagging", default='',
                  help="Optional file of RFI flags in for of [time,freq,corrprod] produced by the workflow maneger (Default = %default)")

(opts, args) = parser.parse_args()

if len(args) ==0:
    raise RuntimeError('Please specify an h5 file to load.')
    


# frequency channels to keep
start_freq_channel = int(opts.freq_keep.split(',')[0])
end_freq_channel = int(opts.freq_keep.split(',')[1])
start_freq_channel = 200
end_freq_channel = 800


h5 = katdal.open(args)
#h5 = katdal.open('1387000585.h5')
fileprefix = os.path.join(opts.output_dir,os.path.splitext(args[0].split('/')[-1])[0])
nice_filename =  fileprefix+ '_antenna_phase_stability'

pp = PdfPages(nice_filename+'.pdf')
for pol in ('h','v'):
    h5.select(channels=slice(start_freq_channel,end_freq_channel),pol=pol,corrprods='cross',scans='track',dumps=slice(1,600)) 
    # loop over both polarisations
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
    gains = stefcal( np.concatenate( (np.mean(vis,axis=0), np.mean(vis.conj(),axis=0)), axis=-1) , N_ants, full_antA, full_antB, num_iters=50,weights=weights)
    calfac = 1./(gains[np.newaxis][:,:,full_antA]*gains[np.newaxis][:,:,full_antB].conj())

    h5.select(channels=slice(start_freq_channel,end_freq_channel),pol=pol,corrprods='cross',scans='track')
    data = np.zeros((h5.shape[0:3:2]),dtype=np.complex)
    i = 0
    for scan in h5.scans():
        print scan
        vis = h5.vis[:,:,:]
        data[i:i+h5.shape[0]] = mean((vis*calfac[:,:,:h5.shape[-1]])[:,flaglist,:],axis=1)
        i += h5.shape[0]
    figlist = []
    figlist += plot_AntennaGain(gains,h5.channel_freqs,h5.inputs)
    fig = plt.figure()
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
        returntext,pltfig = calc_stats(h5.timestamps[:],np.angle(data[:,i]) ,pol="%s,%s"%(ant1,ant2),windowtime=1200,minsamples=1)
        pltfig.savefig(pp,format='pdf') 
        plt.close(pltfig)
        fig = plt.figure(None,figsize = (10,10))
        plt.figtext(0.1,0.1,'\n'.join(returntext),fontsize=10)
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



