#!/usr/bin/python
#Plots uncalibrated power, noise diode firings, derived gains to assess gain stability and effectiveness of the gain calibration
 
import numpy as np
import matplotlib.pyplot as plt   
import optparse
import os.path
import sys

import scape
import katpoint

# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <directories or files>",
                               description=" ")
parser.add_option('-a', '--baseline', dest='baseline', type="string", metavar='BASELINE', default='AxAx',
                  help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-n", "--nd_models", dest="nd_models", type="string", default='',
                  help="Name of optional directory containing noise diode model files")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", type="string", default='100,400',
                  help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")
parser.add_option("-m", "--min_nd", dest="min_nd", type="float", default=0,
                  help="minimum samples of noise diode to use for calibration")
parser.add_option("-t", "--time_width", dest="time_width", type ="float", default=240, 
                  help="time-width over which to smooth the gains")
parser.add_option("-o", "--output", dest="outfilebase", type="string", default='gain_stability',
                  help="Base name of output files (*.jpg for plots)")

(opts, args) = parser.parse_args()

if len(args) ==0:
    print 'Please specify an h5 file to load.'
    sys.exit(1)


# frequency channels to keep
start_freq_channel = int(opts.freq_keep.split(',')[0])
end_freq_channel = int(opts.freq_keep.split(',')[1])

d = scape.DataSet(args[0], baseline=opts.baseline, nd_models=opts.nd_models)
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

def plot_figures(d_uncal, d_cal, time, gain_hh, gain_vv):
   plt.figure()
   plot_filename=opts.outfilebase+ 'HH.jpg'
   plt.clf()
   F=plt.gcf()
   F.set_size_inches(8,12)
   plt.subplots_adjust(hspace=0.4)
   ax1 = plt.subplot(611)
   scape.plot_xyz(d_uncal.select(flagkeep='~nd_on', copy=False),'time','amp', pol='HH', add_breaks=False, labels=[], power_in_dB = False, compact = False)
   ax2 = plt.subplot(612)
   plt.plot(np.hstack(time.data),gain_hh, marker = '.', label='HH', linewidth=0)
   plt.xlim(tmin,tmax)
   plt.xlabel(time.label)
   plt.ylabel('Gain HH')
   ax3 = plt.subplot(613)
   scape.plot_xyz(d_cal.select(flagkeep='~nd_on', copy=False),'time','amp', pol='HH', add_breaks=False, labels=[], power_in_dB = False, compact = False)
   ax4 = plt.subplot(614)
   scape.plot_xyz(d_uncal.select(flagkeep='nd_on', copy=False),'time','amp', pol='HH', add_breaks=False, labels=[], power_in_dB = False, compact = False)
   ax5 = plt.subplot(615)
   scape.plot_xyz(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'az', add_breaks=False, labels=[], compact = False)
   ax6 = plt.subplot(616)
   scape.plot_xyz(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'el', add_breaks=False, labels=[], compact = False)
   #plt.savefig(plot_filename, dpi=600,bbox_inches='tight')
   plt.show()
 
   plt.figure()
   plot_filename= opts.outfilebase+'VV.jpg'
   plt.clf()
   F=plt.gcf()
   F.set_size_inches(8,12)
   plt.subplots_adjust(hspace=0.4)
   ax1 = plt.subplot(611)
   scape.plot_xyz(d_uncal.select(flagkeep='~nd_on', copy=False),'time','amp', pol='VV', add_breaks=False, labels=[], power_in_dB = False, compact = False)
   ax2 = plt.subplot(612)
   plt.plot(np.hstack(time.data),gain_vv, marker = '.', label='VV', linewidth=0)
   plt.xlabel(time.label)
   plt.xlim(tmin,tmax)
   plt.ylabel('Gain VV')
   ax3 = plt.subplot(613)
   scape.plot_xyz(d_cal.select(flagkeep='~nd_on', copy=False),'time','amp', pol='VV', add_breaks=False, labels=[], power_in_dB = False, compact = False)
   ax4 = plt.subplot(614)
   scape.plot_xyz(d_uncal.select(flagkeep='nd_on', copy=False),'time','amp', pol='VV', add_breaks=False, labels=[], power_in_dB = False, compact = False)
   ax5 = plt.subplot(615)
   scape.plot_xyz(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'az', add_breaks=False, labels=[], compact = False)
   ax6 = plt.subplot(616)
   scape.plot_xyz(d_uncal.select(flagkeep='~nd_on', copy=False), 'time', 'el', add_breaks=False, labels=[], compact = False)
   #plt.savefig(plot_filename, dpi=600, bbox_inches='tight')
   plt.show()

plot_figures(d_uncal, d_cal, time, gain_hh, gain_vv)

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

print("Fitting a first order polynomial to amplitude data.")
print("mean value of HH amplitude is: %.3e"%amp_hh.mean())
print("Std. dev of HH amplitude is: %.3e"%amp_detrend_hh.std())
print("Percentage variation is: %.3f"%(amp_detrend_hh.std()/amp_hh.mean()*100))
print("mean value of VV amplitude is: %.3e"%amp_vv.mean())
print("Std. dev of VV amplitude is: %.3e"%amp_detrend_vv.std())
print("Percentage variation is: %.3f"%(amp_detrend_vv.std()/amp_vv.mean()*100))


