#!/usr/bin/python
#Plots uncalibrated power, noise diode firings, derived gains to assess
#gain stability and effectiveness of the gain calibration
import numpy as np
import matplotlib.pyplot as plt
import optparse
import katdal
from matplotlib.backends.backend_pdf import PdfPages
import pandas
import os
from katsdpscripts import git_info
import pickle
import h5py
from katsdpcal import calprocs



def read_and_select_file(data, flags_file=None, value=np.inf):
    """
    Read in the input h5 file and make a selection based on kwargs.
    data : katdal object
    flags_file:  {string} filename of h5 flagfile to open

    Returns:
        A array with the visibility data and bad data changed to {value}.
    """

     #Check there is some data left over
    if data.shape[0] == 0:
        raise ValueError('No data to process.')

    if flags_file is None or flags_file == '':
        print('No flag data to process. Using the file flags')
        file_flags = data.flags[:]
    else:
        #Open the flags file
        ff = h5py.File(flags_file)
        #Select file flages based on h5 file selection
        file_flags = ff['flags'].value
        file_flags = file_flags[data.dumps]
        file_flags = file_flags[:, data._freq_keep]
        file_flags = file_flags[:, :, data._corrprod_keep]
        #Extend flags
        #flags = np.sum(file_flags,axis=-1)
    return np.ma.masked_array(data.vis[:], mask=file_flags, fill_value=value)


def polyfitstd(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Modified polyfit:Any nan values in x are 'masked' and std is returned .
    """
    if np.isnan(y).sum() >= y.shape[0] + 1:
        return 0.0
    z = np.ma.array(data=np.nan_to_num(y), mask=np.isnan(y))
    if x.shape[0] <= deg + 1:
        z = np.zeros((deg + 1))
        z[-1] = np.ma.mean(x)
        return z[0]
    gg = np.ma.polyfit(x, z, deg, rcond, full, w, cov)
    #if np.isnan(gg[0]) : raise RuntimeError('NaN in polyfit, Error')
    return anglestd(z - x * gg[0])


def rolling_window(a, window, axis=-1, pad=False, mode='reflect', **kargs):
    """
        This function produces a rolling window shaped data with 
        the rolled data in the last col
        a      :  n-D array of data
        window : integer is the window size
        axis   : integer, axis to move the window over
        default is the last axis.
        pad    : {Boolean} Pad the array to the origanal size
        mode : {str, function} from the function numpy.pad
        One of the following string values or a user supplied function.
        'constant'      Pads with a constant value.
        'edge'          Pads with the edge values of array.
        'linear_ramp'   Pads with the linear ramp between end_value and the
        array edge value.
        'maximum'       Pads with the maximum value of all or part of the
        vector along each axis.
        'mean'          Pads with the mean value of all or part of the
        con  vector along each axis.
        'median'        Pads with the median value of all or part of the
        vector along each axis.
        'minimum'       Pads with the minimum value of all or part of the
        vector along each axis.
        'reflect'       Pads with the reflection of the vector mirrored on
        the first and last values of the vector along each
        axis.
        'symmetric'     Pads with the reflection of the vector mirrored
        along the edge of the array.
        'wrap'          Pads with the wrap of the vector along the axis.
        The first values are used to pad the end and the
        end values are used to pad the beginning.
        <function>      of the form padding_func(vector, iaxis_pad_width,
        iaxis, **kwargs)
        see numpy.pad notes
        **kargs are passed to the function numpy.pad

        Returns:
        an array with shape = np.array(a.shape+(window,))
        and the rolled data on the last axis

        Example:
        import numpy as np
        data = np.random.normal(loc=1,
            scale=np.sin(5*np.pi*np.arange(10000).astype(float)/10000.)+1.1,
            size=10000)
        stddata = rolling_window(data, 400).std(axis=-1)
        """
    if axis == -1 :
        axis = len(a.shape)-1
    if pad :
        pad_width = []
        for i in xrange(len(a.shape)):
            if i == axis:
                pad_width += [(window // 2, window // 2 -1 +np.mod(window, 2))]
            else :
                pad_width += [(0, 0)]
        a = np.pad(a, pad_width=pad_width, mode=mode, **kargs)
    a1 = np.swapaxes(a, axis, -1) # Move target axis to last axis in array
    shape = a1.shape[:-1] + (a1.shape[-1] - window + 1, window)
    strides = a1.strides + (a1.strides[-1], )
    return np.lib.stride_tricks.as_strided(a1, shape=shape, strides=strides).swapaxes(-2, axis) # Move original axis to


def model(x, a, m, c):
    a = 1.
    return np.sqrt(a**2)*np.exp(1j*(m*x))*np.exp(1j*(np.sqrt(c**2)))

def residuals(params, w, Z):
    R, C, L = params
    diff = model(w, R, C, L) - Z
    return diff.real**2 + diff.imag**2 # np.abs(diff)#np.angle(diff) #


def v_detrend(x):
    result = np.zeros((x.shape[0], 3))
    for i in xrange(x.shape[0]) :
        if i%200 == 0 :print(" %i of %i"%(i,x.shape[0]) )
        result[i,:] = fit_phase_std(np.arange(x.shape[-1]),x[i,:])
    return result


def detrend(x):
    return fit_phase_std(np.arange(x.shape[0]), x)

def cplot(data, *args, **kwargs):
    if data.dtype.kind == 'c':
        plt.plot(np.real(data), np.imag(data), *args, **kwargs)
    else : plt.plot(data, *args, **kwargs)

def mean(a, axis=None):
    """This function calclates the mean along the chosen axis of the array
    This function has been writen to calculate the mean 
    of complex numbers correctly by taking the mean of 
    the argument & the angle (exp(1j*theta) )
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
        r = np.ma.sqrt(a.real ** 2 + a.imag ** 2).mean(axis=axis)
        th = np.ma.arctan2(a.imag, a.real)
        sa = (np.ma.sin(th)).sum(axis=axis)
        ca = (np.ma.cos(th)).sum(axis=axis)
        thme = np.ma.arctan2(sa, ca)
        return r*np.ma.exp(1j * thme)
    else:
        return np.mean(a, axis=axis)


def yamartino_method(a, axis=None):
    """This function calclates the standard devation along the
    chosen axis of the array. This function has been writen to
    calculate the mean of complex numbers correctly by taking
    the standard devation of the argument & the
    angle (exp(1j*theta) ). This uses the Yamartino method
    which is a one pass method of estimating the standard
    devation of an angle. 
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real ** 2 + a.imag ** 2).std(axis=axis)#mean
        th = np.arctan2(a.imag,a.real)
        if axis is None :
            sa = (np.sin(th) / len(th)).sum()
            ca = (np.cos(th) / len(th)).sum()
        else:
            sa = (np.sin(th) / len(th)).sum(axis=axis)
            ca = (np.cos(th) / len(th)).sum(axis=axis)
        e = np.sqrt(1. - (sa ** 2 + ca ** 2))
        thsd = np.arcsin(e)*(1. + (2. / np.sqrt(3) - 1.) * e ** 3)
        return r * np.exp(1j * thsd)
    else:
        return np.std(a, axis=axis)

def std(a, axis=None):
    """This function calclates the standard devation along the
    chosen axis of the array. This function has been writen to
    calculate the mean of complex numbers correctly by taking
    the standard devation of the argument & the 
    angle (exp(1j*theta) ).
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
    The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    if a.dtype.kind == 'c':
        r = np.sqrt(a.real ** 2 + a.imag ** 2).std(axis=axis)#mean
        th = np.arctan2(a.imag, a.real)
        sa = np.sin(th).sum(axis=axis)
        ca = np.cos(th).sum(axis=axis)
        thme = np.arctan2(sa, ca)
        nshape = np.array(th.shape)
        nshape[axis] = 1
        S0 = 1 - np.cos(th - np.reshape(thme, nshape)).mean(axis=axis)
        return r * np.exp(1j * np.sqrt(-2. * np.log(1. - S0)))
    else:
        return np.std(a, axis=axis)


def absstd(a, axis=None):
    """This function calclates the standard devation along the
    chosen axis of the array. This function has been writen to
    calculate the mean of complex numbers correctly by taking
    the standard devation  of the angle (exp(1j*theta) )
    and standard devation over the mean of the argument
    Input :
    a    : N-D numpy array
    axis : The axis to perform the operation over
           The Default is over all axies
    Output:
         This returns a an array or a one value array
    Example:
    """
    a = np.ma.array(data=np.nan_to_num(a), mask=np.isnan(a))
    if a.dtype.kind == 'c':
        rstd = np.ma.sqrt(a.real ** 2 + a.imag ** 2).std(axis=axis)#std
        rmean = np.ma.sqrt(a.real ** 2 + a.imag ** 2).mean(axis=axis)
        th = np.ma.arctan2(a.imag, a.real)
        sa = np.ma.sin(th).sum(axis=axis)
        ca = np.ma.cos(th).sum(axis=axis)
        thme = np.ma.arctan2(sa, ca)
        nshape = np.array(th.shape)
        nshape[axis] = 1
        S0 = 1 - np.ma.cos(th - np.ma.reshape(thme, nshape)).mean(axis=axis)
        return (rstd / rmean) * np.exp(1j * np.ma.sqrt( - 2. * np.log(1. - S0)))
    else:
        return np.std(a,axis=axis)/np.mean(a,axis=axis)

def anglestd(a, axis=None):
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
    a = np.exp(1j * a)
    #rstd = np.sqrt(a.real**2 + a.imag**2).std(axis=axis)#std
    #rmean = np.sqrt(a.real**2 + a.imag**2).mean(axis=axis)
    th = np.arctan2(a.imag, a.real)
    sa = np.sin(th).sum(axis=axis)
    ca = np.cos(th).sum(axis=axis)
    thme = np.arctan2(sa, ca)
    nshape = np.array(th.shape)
    nshape[axis] = 1
    S0 = 1 - np.cos(th - np.reshape(thme, nshape)).mean(axis=axis)
    return np.angle(np.exp(1j * np.sqrt( - 2. * np.log(1. - S0))))

def anglemax(x):
    x = np.exp(1j * x)
    #x = np.ma.array(data=np.nan_to_num(x),mask=x.mask +np.isnan(x))
    return np.angle(x / mean(x)).max()

def anglemin(x):
    x = np.exp(1j * x)
    #x = np.ma.array(data=np.nan_to_num(x),mask=x.mask +np.isnan(x))
    return np.angle(x / mean(x)).min()

def angle_mean(x):
    x = np.exp(1j * x)
    #x = np.ma.array(data=np.nan_to_num(x),mask=x.mask +np.isnan(x))
    return np.angle(mean(x))


def peak2peak(x):
    #x = np.exp(1j* x)
    #x = np.ma.array(data=np.nan_to_num(x),mask=x.mask +np.isnan(x))
    return anglemax(x) - anglemin(x)

def angle_std(a, axis=None):
    """This function calclates the standard devation along
    the chosen axis of the array. This function has been
    writen to calculate the mean of complex numbers correctly
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
    a = np.exp(1j * a)
    #rstd = np.sqrt(a.real**2 + a.imag**2).std(axis=axis)#std
    #rmean = np.sqrt(a.real**2 + a.imag**2).mean(axis=axis)
    th = np.arctan2(a.imag, a.real)
    sa = np.sin(th).sum(axis=axis)
    ca = np.cos(th).sum(axis=axis)
    thme = np.arctan2(sa, ca)
    nshape = np.array(th.shape)
    nshape[axis] = 1
    S0 = 1 - np.cos(th-np.reshape(thme, nshape)).mean(axis=axis)
    #print a.imag,a.real,th,sa,ca,thme,nshape,S0,np.degrees(np.angle(np.exp(1j*np.sqrt(-2.*np.log(1.-S0)))))
    return np.angle(np.exp(1j * np.sqrt(- 2. * np.log(1. - S0))))

def fit_phase_std(x, y):
    """
    fit a 3 component model to the data to remove phase and fringe rate and amplitude
    Any nan values in x are 'masked' and std is returned .
    """
    if np.isnan(y).sum() >= y.shape[0] + 1 :
        return 0.0 # All Nan
    if (~np.isnan(y)).sum() <= 3 :
        return 0.0 # Not enough data
    x_m = x - x.mean()
    #y = np.angle(np.exp(1j*np.angle(y))/mean(y))
    m= ((x_m) * (y - y.mean())).sum() / ((x_m)** 2).sum()
    #print "m=",m
    #p_guess = [1.,0.,0. ]
    #params, cov = optimize.leastsq(residuals, p_guess, args=(x, z))
    # print params, cov
    #if np.isnan(gg[0]) : raise RuntimeError('NaN in polyfit, Error')
    #np.exp(1j*np.angle(y))/(np.exp(1j*(m*x))
    #plot(y-(m*x))
    return  anglestd(y - (m * x))

def calc_stats(timestamps, gain, pol='no polarizarion', windowtime=1200, minsamples=1200):
    """ calculate the Stats needed to evaluate the observation"""
    returntext = []
    #note gain is in radians
    #change_el = pandas.rolling_apply(offset_el_ts,window=4*60/6.,min_periods=0,func=calc_change,freq='360s')*3600

    gain_ts = pandas.Series(np.angle(gain), pandas.to_datetime(timestamps, unit='s'))

    #window_occ = pandas.rolling_count(gain_ts,windowtime)/float(windowtime)
    #full = np.where(window_occ==1)
    #note std is returned in degrees
    std = (pandas.rolling_apply(gain_ts,window=windowtime,func=angle_std,min_periods=minsamples))
    peakmin= ((pandas.rolling_apply(gain_ts,window=windowtime,func=anglemin,min_periods=minsamples)))
    peakmax= ((pandas.rolling_apply(gain_ts,window=windowtime,func=anglemax,min_periods=minsamples)))
    gain_val_corr = ((pandas.rolling_apply(gain_ts,window=windowtime,func=angle_mean,min_periods=minsamples)))
    #gain_val = pandas.Series(gain_ts-gain_val_corr, pandas.to_datetime(timestamps, unit='s') )
    gain_val = pandas.Series(np.angle(np.exp(1j*gain_ts)/np.exp(1j*gain_val_corr)), pandas.to_datetime(timestamps, unit='s'))

    peak =  ((pandas.rolling_apply(gain_ts,window=windowtime,func=peak2peak,min_periods=minsamples)))
    dtrend_std = (pandas.rolling_apply(gain_ts,window=windowtime,func=detrend,min_periods=minsamples))
    #trend_std = pandas.rolling_apply(ts,5,lambda x : np.ma.std(x-(np.arange(x.shape[0])*np.ma.polyfit(np.arange(x.shape[0]),x,1)[0])),1)
    timeval = timestamps.max()-timestamps.min()


    #rms = np.sqrt((gain**2).mean())
    returntext.append("Total time of observation : %f (seconds) with %i accumulations."%(timeval,timestamps.shape[0]))
    #returntext.append("The mean gain of %s is: %.5f"%(pol,gain.mean()))
    #returntext.append("The Std. dev of the gain of %s is: %.5f"%(pol,gain.std()))
    #returntext.append("The RMS of the gain of %s is : %.5f"%(pol,rms))
    #returntext.append("The Percentage variation of %s is: %.5f"%(pol,gain.std()/gain.mean()*100))
    returntext.append("The mean Peak to Peak range over %i seconds of %s is: %.5f (req < 13 )  "%(windowtime,pol,np.degrees(peak.mean())))
    returntext.append("The Max Peak to Peak range over %i seconds of %s is: %.5f  (req < 13 )  "%(windowtime,pol,np.degrees(peak.max())) )
    returntext.append("The mean variation over %i seconds of %s is: %.5f    "%(windowtime,pol,np.degrees(std.mean())) )
    returntext.append("The Max  variation over %i seconds of %s is: %.5f    "%(windowtime,pol,np.degrees(std.max())) )
    returntext.append("The mean detrended variation over %i seconds of %s is: %.5f    (req < 2.3 )"%(windowtime,pol,np.degrees(dtrend_std.mean())))
    returntext.append("The Max  detrended variation over %i seconds of %s is: %.5f    (req < 2.3 )"%(windowtime,pol,np.degrees(dtrend_std.max())))
    pltobj = plt.figure(figsize=[11,20])

    plt.suptitle(h5.name)
    plt.subplots_adjust(bottom=0.15, hspace=0.35, top=0.95)
    plt.subplot(311)
    plt.title('phases for '+pol)
    (gain_val* 180./np.pi).plot(label='phase( - rolling mean)')
    (peakmax * 180./np.pi).plot(label='rolling max')
    (peakmin * 180./np.pi).plot(label='rolling min')
    plt.legend(loc='best')
    plt.ylabel('Gain phase (deg)')

    ax2 = plt.subplot(312)
    plt.title('Peak to peak variation of %s, %i Second sliding Window'%(pol,windowtime,))
    (peak* 180./np.pi).plot(color='blue')
    ax2.axhline(13,ls='--', color='red')
    #plt.legend(loc='best')
    plt.ylabel('Variation (deg)')

    ax3 = plt.subplot(313)
    plt.title('Detrended Std of %s, %i Second sliding Window'%(pol,windowtime,))
    (std* 180./np.pi).plot(color='blue',label='Std')
    (dtrend_std* 180./np.pi).plot(color='green',label='Detrended Std')
    ax3.axhline(2.2,ls='--', color='red')
    plt.legend(loc='best')
    plt.ylabel('Variation (deg)')
    plt.xlabel('Date/Time')

    pltobj2 = plt.figure(figsize=[11,11])
    plt.suptitle(h5.name)
    plt.subplots_adjust(bottom=0.15, hspace=0.35, top=0.95)
    plt.subplot(111)
    plt.title('Raw phases for '+pol)
    (gain_ts* 180./np.pi).plot(label='Raw phase')
    plt.legend(loc='best')
    plt.ylabel('Phase (deg)')
    plt.xlabel('Date/Time')
    plt.legend(loc='best')
    plt.figtext(0.89, 0.05, git_info(), horizontalalignment='right',fontsize=10)

    return returntext,pltobj,pltobj2  # a plot would be cool



# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description=" This produces a pdf file with graphs decribing the gain stability for each antenna in the file")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", type="string", default='211,3896',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")
parser.add_option("-o","--output_dir", default='.', help="Output directory for pdfs. Default is cwd")
parser.add_option("-c", "--channel-mask", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle',
                  help="Optional pickle file with boolean array specifying channels to mask (Default = %default)")
parser.add_option("-r", "--rfi-flagging", default='',
                  help="Optional file of RFI flags in for of [time,freq,corrprod] produced by the workflow maneger (Default = %default)")
parser.add_option( '--ref', dest='ref_ant',  default=None,help="Reference antenna, default is first antenna in the python dictionary")

(opts, args) = parser.parse_args()

if len(args) ==0:
    raise RuntimeError('Please specify an h5 file to load.')


output_dir = '.'

h5 = katdal.open(args[0],ref_ant=opts.ref_ant) 
ref_ant_ind = [ant.name for ant in h5.ants].index(h5.ref_ant)
n_chan = np.shape(h5.channels)[0]
if not opts.freq_keep is None :
    start_freq_channel = int(opts.freq_keep.split(',')[0])
    end_freq_channel = int(opts.freq_keep.split(',')[1])
    edge = np.tile(True, n_chan)
    edge[slice(start_freq_channel, end_freq_channel)] = False
else :
    edge = np.tile(False, n_chan)
#load static flags if pickle file is given
channel_mask ='/var/kat/katsdpscripts/RTS/rfi_mask.pickle'
rfi_flagging = ''
if len(channel_mask)>0:
    pickle_file = open(channel_mask)
    rfi_static_flags = pickle.load(pickle_file)
    pickle_file.close()
    if n_chan > rfi_static_flags.shape[0] :
        rfi_static_flags = rfi_static_flags.repeat(8) # 32k mode
else:
    rfi_static_flags = np.tile(False, n_chan)
static_flags = np.logical_or(edge,rfi_static_flags)
fileprefix = os.path.join(opts.output_dir,os.path.splitext(args[0].split('/')[-1])[0])
nice_filename =  fileprefix+ '_antenna_phase_stability'
pp = PdfPages(nice_filename+'.pdf')

for pol in ('h','v'):
    h5.select(channels=~static_flags,pol=pol,scans='track')
    h5.antlist = [a.name for a in h5.ants]
    h5.bls_lookup = calprocs.get_bls_lookup(h5.antlist,h5.corr_products)
    data = np.ma.zeros((h5.shape[0],len(h5.ants)),dtype=np.complex)
    i = 0
    for scan in h5.scans():
        vis = read_and_select_file(h5, flags_file=rfi_flagging)
        print "Read data: %s:%i target:%s   (%i samples)"%(scan[1],scan[0],scan[2].name,vis.shape[0])
        bl_ant_pairs = calprocs.get_bl_ant_pairs(h5.bls_lookup)
        antA, antB = bl_ant_pairs
        cal_baselines = vis.mean(axis=1) 
                         #/(bandpass[np.newaxis,:,antA[:len(antA)//2]]*np.conj(bandpass[np.newaxis,:,antB[:len(antB)//2]]))[:,:,:]).mean(axis=1)
        data[i:i+h5.shape[0],:] = calprocs.g_fit(cal_baselines[:,:],h5.bls_lookup,refant=ref_ant_ind)
        #data.mask[i:i+h5.shape[0],:] =  # this is for when g_fit handels masked arrays
        print "Calculated antenna gain solutions for %i antennas with ref. antenna = %s "%(data.shape[1],h5.ref_ant)
        i += h5.shape[0]

    fig = plt.figure()
    plt.suptitle(h5.name)
    plt.title('Phase angle in Antenna vs. Time for %s pol  '%(pol))
    plt.xticks( np.arange(len(h5.antlist)), h5.antlist ,rotation='vertical')
    plt.imshow(np.degrees(np.angle(data)),aspect='auto',interpolation='none')
    plt.ylabel('Time, (colour angle in degrees)');plt.xlabel('Antenna')
    plt.colorbar()
    fig.savefig(pp,format='pdf')
    plt.close(fig)

    for i,ant in  enumerate(h5.antlist):
        print "Generating Stats on the Antenna %s"%(ant)
        #mask = ~data.mask[:,i] # this is for when g_fit handels masked arrays
        mask = slice(0,data.shape[0])
        returntext,pltfig,pltfig2 = calc_stats(h5.timestamps[mask],data[mask,i].data ,pol="%s,%s"%(ant,pol),windowtime=1200//4,minsamples=1200//4)
        pltfig.savefig(pp,format='pdf')
        plt.close(pltfig)
        pltfig2.savefig(pp,format='pdf')
        plt.close(pltfig2)
        fig = plt.figure(None,figsize = (10,10))
        plt.figtext(0.1,0.5,'\n'.join(returntext),fontsize=10)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
pp.close()
plt.close('all')
