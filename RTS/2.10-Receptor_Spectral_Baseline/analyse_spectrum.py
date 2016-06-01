#!/usr/bin/python

import optparse
from katsdpscripts.RTS import spectral_baseline

def parse_arguments():
    parser = optparse.OptionParser(usage="%prog [opts] <file>")
    parser.add_option("-p", "--polarisation", type="string", default="HH,VV", help="List of polarisation to produce spectra of, options are I, HH, VV, HV, VH. Default is I.")
    parser.add_option("-b", "--baseline", type="string", default=None, help="Baseline to load (e.g. 'ant1,ant1' for antenna 1 auto-corr), default is first single-dish baseline in file.")
    parser.add_option("-t", "--target", type="string", default=None, help="Target to plot spectrum of, default is the first target in the file.")
    parser.add_option("-c", "--freqaverage", type="float", default=1.0, help="Frequency averaging interval in MHz. Default is 1 MHz.")
    parser.add_option("-d", "--timeaverage", type="float", default=60.0, help="Time averageing interval in minutes. Default is 60 minutes.")
    parser.add_option("-f", "--freq-chans", help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 90% of the bandpass.")
    parser.add_option("-o","--output_dir", default='.', help="Output directory for pdfs. Default is cwd")
    parser.add_option("-l","--flags-file", default=None, help="Name of h5 file containing flags (output by rfi_report.py)")
    parser.add_option("--correct", default='spline', help="Method to use to correct the spectrum in each average timestamp. Options are 'spline', 'median', 'gaussian'")
    parser.add_option("-s","--smooth", default=3, help="Smoothing factor for --correct method (this is approximately the number of channels over which to smooth)")
    (opts, args) = parser.parse_args()

    return opts, args

opts, args = parse_arguments()
spectral_baseline.analyse_spectrum(args[0],output_dir=opts.output_dir,polarisation=opts.polarisation,baseline=opts.baseline,target=opts.target,
            freqav=opts.freqaverage,timeav=opts.timeaverage,freq_chans=opts.freq_chans,correct=opts.correct,flags_file=opts.flags_file)