#!/usr/bin/python
#Plots uncalibrated power, noise diode firings, derived gains to assess gain stability and effectiveness of the gain calibration

import numpy as np
import matplotlib.pyplot as plt
import optparse
import scape
import katdal
from matplotlib.backends.backend_pdf import PdfPages
import pandas

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
    if np.isnan(gg[0]) : raise RuntimeError('NaN in polyfit, Error')
    return np.ma.std(z-x*gg[0])

def detrend(x):
    return polyfitstd(np.arange(x.shape[0]),x,1)


def plot_figures(d_uncal, d_cal, time, gain,pol):
    """ This function plots the six graphs for A polarization
    This takes in two scape  data sets:
        d_uncal : scape dataset that contains the averaged uncalibrated data
        d_cal   : scape dataset that contains the  averaged calibraded data in Kelvin
        time    : np.array of timestamps of scans in that data set
        gain    : np.array of the gains for the polarization
        pol     : string polorization code eg. 'HH'
    Returns :
         matplotlib figure object of the graph produced
    """
    fig = plt.figure()
    plt.title(pol)
    plt.clf()
    F=plt.gcf() # fig ?
    F.set_size_inches(8,12)
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(611)
    scape.plot_xyz(d_uncal.select(flagkeep='~nd_on', copy=False),'time','amp', pol=pol, add_breaks=False, labels=[], power_in_dB = False, compact = False)
    plt.subplot(612)
    plt.plot(np.hstack(time.data),gain_hh, marker = '.', label=pol, linewidth=0)
    plt.xlim(tmin,tmax)
    plt.xlabel(time.label)
    plt.ylabel('Gain %s'%(pol,))
    plt.subplot(613)
    scape.plot_xyz(d_cal.select(flagkeep='~nd_on', copy=False),'time','amp', pol=pol, add_breaks=False, labels=[], power_in_dB = False, compact = False)
    plt.subplot(614)
    scape.plot_xyz(d_uncal.select(flagkeep='nd_on', copy=False),'time','amp', pol=pol, add_breaks=False, labels=[], power_in_dB = False, compact = False)
    plt.subplot(615)
    scape.plot_xyz(d_uncal.select(copy=False), 'time', 'az', add_breaks=False, labels=[], compact = False) # select tracks ?
    plt.subplot(616)
    scape.plot_xyz(d_uncal.select(copy=False), 'time', 'el', add_breaks=False, labels=[], compact = False)
    #plt.savefig(plot_filename, dpi=600,bbox_inches='tight')
    return fig



# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <file>",
                               description=" This produces a pdf file with graphs decribing the gain sability for each antenna in the file")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", type="string", default='200,800',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")
parser.add_option("-m", "--min_nd", dest="min_nd", type="float", default=10,
                  help="minimum samples of noise diode to use for calibration")
parser.add_option("-t", "--time_width", dest="time_width", type ="float", default=240,
                  help="time-width over which to smooth the gains")
parser.add_option("-a", "--ant", dest="ant", type ="str", default='',
                  help="The antenna to examine the gain stability on")

(opts, args) = parser.parse_args()

if len(args) ==0:
    raise RuntimeError('Please specify an h5 file to load.')



# frequency channels to keep
start_freq_channel = int(opts.freq_keep.split(',')[0])
end_freq_channel = int(opts.freq_keep.split(',')[1])

def rolling_window(a, window):
    """ From http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
     This function produces a rowling window shaped data
        a :  1-D array of data
        window : integer is the window size
    Returns:
        an array shape= (N,window) where
        the origanal data is length of N

    Example:
        import numpy as np
        data = np.random.normal(loc=1,scale=np.sin(5*np.pi*np.arange(10000).astype(float)/10000.)+1.1, size=10000)
        stddata = rolling_window(data, 400).std(axis=1)
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def calc_stats(timestamps,gain,pol='no polarizarion',windowtime=1200,minsamples=1):
    """ calculate the Stats needed to evaluate the obsevation"""
    returntext = []
    gain_ts = pandas.Series(gain, pandas.to_datetime(np.round(timestamps), unit='s'))#.asfreq(freq='1s')
    mean = pandas.rolling_mean(gain_ts,windowtime,minsamples)
    std = pandas.rolling_std(gain_ts,windowtime,minsamples)
    windowgainchange = std/mean*100
    dtrend_std = pandas.rolling_apply(gain_ts,windowtime,detrend,minsamples)
    #trend_std = pandas.rolling_apply(ts,5,lambda x : np.ma.std(x-(np.arange(x.shape[0])*np.ma.polyfit(np.arange(x.shape[0]),x,1)[0])),1)
    detrended_windowgainchange = dtrend_std/mean*100
    timeval = timestamps.max()-timestamps.min()
    window_occ = pandas.rolling_count(gain_ts,windowtime)/float(windowtime)

    #rms = np.sqrt((gain**2).mean())
    returntext.append("Total time of obsevation : %f (seconds) with %i accumulations."%(timeval,timestamps.shape[0]))
    #returntext.append("The mean gain of %s is: %.5f"%(pol,gain.mean()))
    #returntext.append("The Std. dev of the gain of %s is: %.5f"%(pol,gain.std()))
    #returntext.append("The RMS of the gain of %s is : %.5f"%(pol,rms))
    #returntext.append("The Percentage variation of %s is: %.5f"%(pol,gain.std()/gain.mean()*100))
    returntext.append("The mean Percentage variation over %i seconds of %s is: %.5f    (req < 2 )"%(windowtime,pol,windowgainchange.mean()))
    returntext.append("The Max  Percentage variation over %i seconds of %s is: %.5f    (req < 2 )"%(windowtime,pol,windowgainchange.max()))
    returntext.append("The mean detrended Percentage variation over %i seconds of %s is: %.5f    (req < 2 )"%(windowtime,pol,detrended_windowgainchange.mean()))
    returntext.append("The Max  detrended Percentage variation over %i seconds of %s is: %.5f    (req < 2 )"%(windowtime,pol,detrended_windowgainchange.max()))
    #a - np.round(np.polyfit(b,a.T,1)[0,:,np.newaxis]*b + np.polyfit(b,a.T,1)[1,:,np.newaxis])

    pltobj = plt.figure()
    plt.title('Percentage Variation of %s pol, %i Second sliding Window'%(pol,windowtime,))
    windowgainchange.plot(label='Orignal')
    detrended_windowgainchange.plot(label='Detrended')
    window_occ.plot(label='Window Occupancy')
    plt.hlines(2, timestamps.min(), timestamps.max(), colors='k')
    plt.ylabel('Percentage Variation')
    plt.xlabel('Date/Time')
    plt.legend(loc='best')
    #plt.title(" %s pol Gain"%(pol))
    #plt.plot(windowgainchange.mean(),'b',label='20 Min (std/mean)')
    #plt.plot(np.ones_like(windowgainchange.mean())*2.0,'r',label=' 2 level')
    return returntext,pltobj  # a plot would be cool


def remove_rfi(d,width=3,sigma=5,axis=1):
    for i in range(len(d.scans)):
        d.scans[i].data = scape.stats.remove_spikes(d.scans[i].data,axis=axis,spike_width=width,outlier_sigma=sigma)
    return d

gain_hh = np.array(())
gain_vv = np.array(())
timestamps = np.array(())
filename = args[0]
nice_filename =  filename.split('/')[-1]+ '_' +opts.ant+'_gain_stability'
pp = PdfPages(nice_filename+'.pdf')

for filename in args:
    h5 = katdal.open(filename)
    if opts.ant=='' :
        ant= h5.ants[0].name
    else:
        ant = opts.ant
    #h5.select(ants=ant)
    d = scape.DataSet(filename, baseline="%s,%s" % (ant,ant))
    d = d.select(freqkeep=range(start_freq_channel, end_freq_channel+1))
    d = remove_rfi(d,width=21,sigma=5)  # rfi flaging
    #Leave the d dataset unchanged after this so that it can be examined interactively if necessary
    antenna = d.antenna
    d_uncal = d.select(copy = True)
    d_uncal.average()
    d_cal = d.select(copy=True)
    print("do selects")
    #extract timestamps from data
    timestampfile = np.hstack([scan.timestamps for scan in d_cal.scans])
    #get a user-friendly time axis that will plot in the same way as plot_xyz
    time = scape.extract_scan_data(d_cal.scans, 'time')
    tmin = np.min(np.hstack(time.data))
    tmax = np.max(np.hstack(time.data))
    #Get the gain from the noise diodes
    g_hh, g_vv, delta_re_hv, delta_im_hv = scape.gaincal.estimate_gain(d_cal)
    gain_hh = np.r_[gain_hh,g_hh(timestampfile, d.freqs).mean(axis=1)]
    gain_vv = np.r_[gain_vv,g_vv(timestampfile, d.freqs).mean(axis=1)]
    timestamps = np.r_[timestamps,timestampfile]
    print("Applied gains")
    print " gain_hh  %i, gain_vv %i, timestamps %i"%(gain_hh.shape[0],gain_vv.shape[0],timestamps.shape[0])
    #Apply noise diode calibration
    if False:
        d_cal.convert_power_to_temperature(min_samples=opts.min_nd, time_width=opts.time_width)
        d_cal.average()
        fig = plot_figures(d_uncal, d_cal, time, gain_hh, 'HH')
        fig.savefig(pp,format='pdf')
        plt.close()
        fig = plot_figures(d_uncal, d_cal, time, gain_vv, 'VV')
        fig.savefig(pp,format='pdf')
        plt.close()

    #extract data to look at stats
    #time, amp_hh, z =  scape.extract_xyz_data(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'amp', pol = 'HH')
    #time, amp_vv, z =  scape.extract_xyz_data(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'amp', pol = 'VV')
    #time = np.hstack(time.data)
    #amp_hh = np.hstack(amp_hh.data)
    #amp_vv = np.hstack(amp_vv.data)

if True :
    returntext,fig = calc_stats(timestamps,gain_hh,'HH',1200)
    fig.savefig(pp,format='pdf')
    plt.close()
    tmp,fig = calc_stats(timestamps,gain_vv,'VV',1200)
    fig.savefig(pp,format='pdf')
    plt.close()
    returntext += tmp
    #detrend data
    fig = plt.figure(None,figsize = (10,16))
    plt.figtext(0.1,0.1,'\n'.join(returntext),fontsize=10)
    fig.savefig(pp,format='pdf')
pp.close()
plt.close()
###

