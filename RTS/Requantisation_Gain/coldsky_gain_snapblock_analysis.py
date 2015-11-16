#! /usr/bin/python

## Script to evaluate quantisation gain settings using histograms
#  Assumes input file from observation script coldsky_gain_snapshots.py

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
import os,sys
import numpy, string
from scipy.stats import norm
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# ICD Define parameters
N_ADC_BITS   = 10
N_QUANT_BITS = 8

# Read captured snap block data
def read_snapshots(filename):
    # Snap block buffers are simply appended during capture.
    # Both ADC and Quantisation snapblocks will contain 16384 samples,
    # but the quantisation snap block will be a series of channel spectra
    print 'Reading file... '
    try:
        fin = open(datafile, 'r')
        data = fin.readlines()
        fin.close()
    except Exception as E: print E

    print 'Processing file...'
    snap_data = {}
    for line in data:
        data = line.strip().split(':')
        keyword = string.lower(data[0])
        if keyword == 'source': continue # ignore source name -- assume cold sky single target observations
        if keyword == 'gain':
            gain = data[1].strip()
            print "Reading snap block data for gain", gain
            if not snap_data.has_key(gain): snap_data[gain] = {}
            continue
        if keyword == 'antenna':
            antenna = data[1].strip()
            if not snap_data[gain].has_key(antenna):
                snap_data[gain][antenna] = {'adc':{'h':None, 'v':None},
                                            'quant':{'h':None, 'v':None}}
            continue
        if keyword == 'adc h':
            snap_data[gain][antenna]['adc']['h'] = numpy.array(data[1].strip().split(','), dtype=float)
            continue
        if keyword == 'adc v':
            snap_data[gain][antenna]['adc']['v'] = numpy.array(data[1].strip().split(','), dtype=float)
            continue
        if keyword == 'quant h':
            values = []
            values = [numpy.complex(val) for val in data[1].strip().split(',')]
            snap_data[gain][antenna]['quant']['h'] = numpy.array(values)
            continue
        if keyword == 'quant v':
            values = []
            values = [numpy.complex(val) for val in data[1].strip().split(',')]
            snap_data[gain][antenna]['quant']['v'] = numpy.array(values)
            continue
    return snap_data

# Show snap block spectra
def snap_spectra(adc_data, qnt_data, N_samples, N_channels, channel='H', pp=None):
    # Show snap block spectra
    print 'ADC snap block data'
    adc_data = adc_data.reshape(len(adc_data)/N_samples, N_samples)
    adc_band = numpy.zeros_like(adc_data,dtype=numpy.complex)
    for i in xrange(adc_data.shape[0]):
        adc_band[i,:] = numpy.fft.fft(adc_data[i,:])

    print 'Quantisation snap block data'
    qnt_data = qnt_data.reshape(len(qnt_data)/N_channels, N_channels)

    plt.figure(figsize=[13,7])
    plt.subplot(121)
    plt.semilogy((numpy.conj(adc_band[:,0:N_samples/2])*adc_band[:,0:N_samples/2]).real.mean(axis=0))
    plt.axis('tight')
    plt.xlabel('Channels')
    plt.ylabel('Amp [arb dB]')
    plt.title('ADC %s channel spectrum'%channel)
    plt.subplot(122)
    plt.semilogy((numpy.conj(qnt_data)*qnt_data).real.mean(axis=0))
    plt.axis('tight')
    plt.xlabel('Channels')
    plt.ylabel('Amp [arb dB]')
    plt.title('Quant %s channel spectrum'%channel)
    plt.savefig(pp,format='pdf')

# Histogram densities of DBE snap blocks
def build_histograms(data, N_samples, N_channels, min_channel, max_channel, pp=None):
    nr_adc_levels   = 2**N_ADC_BITS
    nr_quant_levels = 2**N_QUANT_BITS
    quant_ratio1=[]
    quant_ratio2=[]
    print 'Generating histograms... '
    for gain in numpy.sort(data.keys()):
        print 'Histograms for gain', gain
        ants = data[gain].keys()
        for ant in ants:
# ADC snap shot if it exists
            if data[gain][ant].has_key('adc'):
                adc_keys = data[gain][ant]['adc'].keys()
                n_adc    = len(adc_keys)
                plt.figure(1)
                plt.clf()
                plt.subplots_adjust(hspace=.7)
                plt.subplots_adjust(wspace=.7)
                for idx in range(n_adc):
                    plt.subplot(1,n_adc,idx+1)
                    histData, bins, patches = plt.hist(data[gain][ant]['adc'][adc_keys[idx]],
                                                       bins = nr_adc_levels,
                                                       range = (-nr_adc_levels/2, nr_adc_levels/2))
                    [mu, sigma] = norm.fit(data[gain][ant]['adc'][adc_keys[idx]])
                    plt.ylabel(r'$\mathrm{Antenna\ %s,\ gain\ %s,\ pol\ %s}$' %(ant, str(gain), adc_keys[idx]))
                    plt.title(r'$\mathrm{ADC:}\ \sigma=%.3f$' %(sigma))
                    plt.axis([bins[numpy.nonzero(histData>0)[0][0]], bins[numpy.nonzero(histData>0)[0][-1]+1], 0, numpy.max(histData)+1.5])
                plt.savefig(pp,format='pdf')

# Quantisation snap shot over bandpass frequency range if it exists
            if data[gain][ant].has_key('quant'):
                quant_keys = data[gain][ant]['quant'].keys()
                n_quant    = len(quant_keys)
                for idx in range(n_quant):
                    quant_data = numpy.array(snap_data[gain][ant]['quant'][quant_keys[idx]])*nr_quant_levels/2
                    quant_data = quant_data.reshape(len(quant_data)/N_channels, N_channels)[:,min_channel:max_channel]
                    [nr,nc] = quant_data.shape
                    quant_data = quant_data.reshape(nr*nc)
                    plt.figure(1)
                    plt.clf()
                    plt.subplots_adjust(hspace=.7)
                    plt.subplots_adjust(wspace=.7)
                    plt.subplot(1,2,1)
                    histData, bins, patches = plt.hist(quant_data.real,
                                                       bins = nr_quant_levels,
                                                       range = (-nr_quant_levels/2, nr_quant_levels/2))
                    [mu, sigma] = norm.fit(quant_data.real)
                    plt.ylabel(r'$\mathrm{Antenna\ %s,\ gain\ %s,\ pol\ %s\ real}$' %(ant, str(gain), quant_keys[idx]))
                    plt.title(r'$\mathrm{Re(quant):}\ \sigma=%.3f$' %(sigma))
                    plt.axis([bins[numpy.nonzero(histData>0)[0][0]], bins[numpy.nonzero(histData>0)[0][-1]+1], 0, numpy.max(histData)+1.5])
                    quant_ratio1.append([gain, numpy.max(histData)/float(numpy.max([histData[numpy.nonzero(histData>0)[0][0]], histData[numpy.nonzero(histData>0)[0][-1]]]))])

                    plt.subplot(1,2,2)
                    histData, bins, patches = plt.hist(quant_data.imag,
                                                       bins = nr_quant_levels,
                                                       range = (-nr_quant_levels/2, nr_quant_levels/2))
                    [mu, sigma] = norm.fit(quant_data.imag)
                    plt.ylabel(r'$\mathrm{Antenna\ %s,\ gain\ %s,\ pol\ %s\ imaginary}$' %(ant, str(gain), quant_keys[idx]))
                    plt.title(r'$\mathrm{Im(quant):}\ \sigma=%.3f$' %(sigma))
                    plt.axis([bins[numpy.nonzero(histData>0)[0][0]], bins[numpy.nonzero(histData>0)[0][-1]+1], 0, numpy.max(histData)+1.5])
                    quant_ratio2.append([gain, float(numpy.max([histData[numpy.nonzero(histData>0)[0][0]], histData[numpy.nonzero(histData>0)[0][-1]]]))/numpy.max(histData)])
                    plt.savefig(pp,format='pdf')

    return [quant_ratio1, quant_ratio2]

# Polynomial fit helper function
def fit_polynomial(x,y,rank=1):
    coeffs = numpy.polyfit(x, y, rank)
    polyfunc= numpy.poly1d(coeffs)
    nx = numpy.linspace(x[0],x[-1], 1000)
    ny = polyfunc(nx)
    return [nx,ny]

if __name__ == '__main__':

    parser = OptionParser(usage='%prog [options] <filename.data>', version="%prog 1.0")
    parser.add_option('--ant',
                      action='store',
                      dest='ant',
                      type=str,
                      default='m063',
                      help='Antenna to use, e.g.  \'m063\'.')
    parser.add_option('--nsamples',
                      action='store',
                      dest='nsamples',
                      type=int,
                      default=16384,
                      help='Largest buffer of ADC raw snap shot output, default = \'%default\'.')
    parser.add_option('--bandwidth',
                      action='store',
                      dest='bandwidth',
                      type=float,
                      default=860e6,
                      help='Observational bandwidth depending on receiver, default = \'%default\' Hz.')
    parser.add_option('--nchannels',
                      action='store',
                      dest='nchannels',
                      type=int,
                      default=4096,
                      help='Number of channels as per observation mode, default = \'%default\'.')
    parser.add_option('--minchn',
                      action='store',
                      dest='minchn',
                      type=int,
                      default=500,
                      help='Min channel number defining bandpass, default = \'%default\'.')
    parser.add_option('--maxchn',
                      action='store',
                      dest='maxchn',
                      type=int,
                      default=3500,
                      help='Max channel number defining bandpass, default = \'%default\'.')
    parser.add_option('-o', '--out',
                      action='store',
                      dest='outfile',
                      type=str,
                      default=None,
                      help='Full path name of output PDF report file.')
    (opts, args) = parser.parse_args()

    if len(args) < 1: raise SystemExit(parser.print_usage())
    datafile = args[0]
    if opts.outfile is None: opts.outfile = os.path.splitext(os.path.basename(datafile))[0]
    else:                    opts.outfile = os.path.splitext(os.path.basename(opts.outfile))[0]
    # Generate output report
    pagetext = 'Pointing of a dish cold sky SCP\n'
    pagetext = pagetext + 'Look at cold sky with attenuated telescope.\n'
    pagetext = pagetext + 'Set a gain value, capture a ADC & quantization snapshots.\n'
    pagetext = pagetext + 'Histogram data, to find a normal distribution.\n'
    pagetext = pagetext + 'Increase gain values while measuring peak to wing height.\n'
    pagetext = pagetext + 'Plot peak to wing ratio over gains to find inflection point in the curve.\n'

    pp = PdfPages(opts.outfile+'.pdf')
    plt.figure()
    plt.axes(frame_on=False)
    plt.xticks([])
    plt.yticks([])
    plt.title("RTS Report %s"%opts.outfile,fontsize=14, fontweight="bold")
    plt.text(0,0,pagetext,fontsize=12)
    plt.savefig(pp,format='pdf')

    snap_data = read_snapshots(datafile)

    gain = numpy.sort(snap_data.keys())[-1]
    print 'Maximum gain of data being investigated =', gain
    adc_data = numpy.array(snap_data[gain][opts.ant]['adc']['h'])
    qnt_data = numpy.array(snap_data[gain][opts.ant]['quant']['h'])
    snap_spectra(adc_data, qnt_data, opts.nsamples, opts.nchannels, pp=pp)
    adc_data = numpy.array(snap_data[gain][opts.ant]['adc']['v'])
    qnt_data = numpy.array(snap_data[gain][opts.ant]['quant']['v'])
    snap_spectra(adc_data, qnt_data, opts.nsamples, opts.nchannels, channel='V', pp=pp)

    [quant_ratio1, quant_ratio2] = build_histograms(snap_data, opts.nsamples, opts.nchannels, min_channel=opts.minchn, max_channel=opts.maxchn, pp=pp)

    print "Finding gain value... "
    np_quant_ratio1 = numpy.array([numpy.array(item, dtype=float) for item in quant_ratio1])
    np_quant_ratio2 = numpy.array([numpy.array(item, dtype=float) for item in quant_ratio2])
    [nx_ratio1,ny_ratio1] = fit_polynomial(np_quant_ratio1[:,0], np_quant_ratio1[:,1]/numpy.max(np_quant_ratio1[:,1]), rank = 2)
    [nx_ratio2,ny_ratio2] = fit_polynomial(np_quant_ratio2[:,0], np_quant_ratio2[:,1]/numpy.max(np_quant_ratio2[:,1]), rank = 2)
    idx = numpy.argmin(numpy.abs(ny_ratio1-ny_ratio2))
    plt.figure()
    plt.hold(True)
    plt.plot(np_quant_ratio1[:,0], np_quant_ratio1[:,1]/numpy.max(np_quant_ratio1[:,1]), 'bo')
    plt.plot(np_quant_ratio2[:,0], np_quant_ratio2[:,1]/numpy.max(np_quant_ratio2[:,1]), 'ro')
    plt.plot(nx_ratio1, ny_ratio1, 'b')
    plt.plot(nx_ratio2, ny_ratio2, 'r')
    plt.axhline(y=numpy.average((ny_ratio1[idx],ny_ratio2[idx])), color='y', linestyle='--')
    plt.axvline(x=numpy.average((nx_ratio1[idx],nx_ratio2[idx])), color='y', linestyle='--')
    plt.title("Coarse measure of gains give gain = %d" % int(numpy.average((nx_ratio1[idx],nx_ratio2[idx]))))
    plt.legend(['peak to wing ratio', 'inverse ratio'],0)
    plt.hold(False)
    plt.savefig(pp,format='pdf')

    pp.close()

# -fin-

