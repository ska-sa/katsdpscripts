#!/usr/bin/python
#Plots uncalibrated power, noise diode firings, derived gains to assess gain stability and effectiveness of the gain calibration
 
import numpy as np
import matplotlib.pyplot as plt   
import optparse
import scape
import katfile
from matplotlib.backends.backend_pdf import PdfPages


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

(opts, args) = parser.parse_args()

if len(args) ==0:
    raise RuntimeError('Please specify an h5 file to load.')
    


# frequency channels to keep
start_freq_channel = int(opts.freq_keep.split(',')[0])
end_freq_channel = int(opts.freq_keep.split(',')[1])

h5 = katfile.open(args[0])
for ant in h5.ants:
    d = scape.DataSet(args[0], baseline="A%sA%s" % (ant.name[3:], ant.name[3:]))
    d = d.select(freqkeep=range(start_freq_channel, end_freq_channel+1))
    #Leave the d dataset unchanged after this so that it can be examined interactively if necessary
    antenna = d.antenna
    d_uncal = d.select(copy = True)
    d_uncal.average()
    d_cal = d.select(copy=True)

    #extract timestamps from data
    timestamps = np.hstack([scan.timestamps for scan in d_cal.scans])
    #get a user-friendly time axis that will plot in the same way as plot_xyz
    time = scape.extract_scan_data(d_cal.scans, 'time')
    tmin = np.min(np.hstack(time.data))
    tmax = np.max(np.hstack(time.data))
    #Get the gain from the noise diodes
    g_hh, g_vv, delta_re_hv, delta_im_hv = scape.gaincal.estimate_gain(d_cal)
    gain_hh = g_hh(timestamps, d.freqs).mean(axis=1)
    gain_vv = g_vv(timestamps, d.freqs).mean(axis=1)

    #Apply noise diode calibration
    d_cal.convert_power_to_temperature(min_samples=opts.min_nd, time_width=opts.time_width)
    d_cal.average()

    nice_filename =  args[0]+ '_' +d.antenna.name+'_gain_stability'
    pp = PdfPages(nice_filename+'.pdf')
    fig = plot_figures(d_uncal, d_cal, time, gain_hh, 'HH')
    fig.savefig(pp,format='pdf') 
    plt.close()
    fig = plot_figures(d_uncal, d_cal, time, gain_vv, 'VV')
    fig.savefig(pp,format='pdf')    
    plt.close()
    #extract data to look at stats
    time, amp_hh, z =  scape.extract_xyz_data(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'amp', pol = 'HH')
    time, amp_vv, z =  scape.extract_xyz_data(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'amp', pol = 'VV')
    time = np.hstack(time.data)
    amp_hh = np.hstack(amp_hh.data)
    amp_vv = np.hstack(amp_vv.data)
    #detrend data
    p_hh = np.polyfit(time,amp_hh,1)
    fit_hh = np.polyval(p_hh,time)
    amp_detrend_hh = amp_hh - fit_hh + amp_hh.mean()
    p_vv = np.polyfit(time,amp_vv,1)
    fit_vv = np.polyval(p_vv,time)
    amp_detrend_vv = amp_vv - fit_vv + amp_vv.mean()
    returntext = []
    returntext.append("Fitting a first order polynomial to amplitude data.")
    returntext.append("mean value of HH amplitude is: %.3e"%amp_hh.mean())
    returntext.append("Std. dev of HH amplitude is: %.3e"%amp_detrend_hh.std())
    returntext.append("Percentage variation is: %.3f"%(amp_detrend_hh.std()/amp_hh.mean()*100))
    returntext.append("mean value of VV amplitude is: %.3e"%amp_vv.mean())
    returntext.append("Std. dev of VV amplitude is: %.3e"%amp_detrend_vv.std())
    returntext.append("Percentage variation is: %.3f"%(amp_detrend_vv.std()/amp_vv.mean()*100))
    fig = plt.figure(None,figsize = (10,16))
    plt.figtext(0.1,0.1,'\n'.join(returntext),fontsize=10)
    fig.savefig(pp,format='pdf')    
    pp.close()
    plt.close()


