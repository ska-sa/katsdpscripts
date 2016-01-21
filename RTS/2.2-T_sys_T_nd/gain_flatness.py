#!/usr/bin/python

import optparse
import katdal
from katsdpscripts.RTS import spectral_baseline as sb
from katsdpscripts.RTS import git_info
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>")
    parser.add_option("-p", "--polarisation", type="string", default="HH,VV", help="List of polarisation to produce spectra of, options are I, HH, VV, HV, VH. Default is I.")
    parser.add_option("-b", "--baseline", type="string", default=None, help="Baseline to load (e.g. 'ant1,ant1' for antenna 1 auto-corr), default is first single-dish baseline in file.")
    parser.add_option("-t", "--target", type="string", default=None, help="Target to plot spectrum of, default is the first target in the file.")
    parser.add_option("-f", "--freq-chans", help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 90% of the bandpass.")
    parser.add_option("-o","--output_dir", default='.', help="Output directory for pdfs. Default is cwd")
    parser.add_option("-l","--flags-file", default=None, help="Name of h5 file containing flags (output by rfi_report.py)")
    (opts, args) = parser.parse_args()

    return opts, args

opts, args = parse_arguments()
input_file = args[0]
h5data = katdal.open(input_file)
baseline = opts.baseline
if opts.target == None:
    target = h5data.catalogue.targets[0].name
else:
    target = opts.target
h5data.select(targets=target, scans='track')
start_time = Time(h5data.timestamps[0], format='unix')
end_time = Time(h5data.timestamps[-1], format='unix')
h5name = h5data.name.split('/')[-1]
output_filename = 'Gain_flatness_'+baseline+'_'+h5name
pdf = PdfPages(output_filename+'.pdf')
fig = plt.figure(figsize=[10,10])
plt.suptitle(h5name+', '+start_time.iso+' - '+end_time.iso)
nplots = len(opts.polarisation.split(','))

for i, pol in enumerate(opts.polarisation.split(',')):
    visdata, weightdata, h5data = \
        sb.read_and_select_file(h5data, bline=baseline, target=target, channels=opts.freq_chans, polarisation=pol, flags_file=opts.flags_file)
    mean_spec = 10*np.log10(visdata.mean(axis=0))
    mean_level = mean_spec.mean() 
    ax = plt.subplot(nplots,1,i)
    plt.title('Gain flatness, '+baseline+pol)
    plt.plot(h5data.channel_freqs/1e6, mean_spec)
    plt.ylabel('Power (dB)')
    plt.xlabel('Frequency (MHz)')
    plt.axhline(mean_level, color='r', alpha=0.5)
    plt.axhline(mean_level+5, color='r')
    plt.axhline(mean_level-5, color='r')
    plt.xlim(h5data.channel_freqs[0]/1e6, h5data.channel_freqs[-1]/1e6)
    plt.grid()
    
plt.figtext(0.5, 0.95, git_info(), horizontalalignment='center',fontsize=10)
fig.savefig(pdf, format='pdf')
pdf.close()