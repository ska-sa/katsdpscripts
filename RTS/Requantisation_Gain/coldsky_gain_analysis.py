#! /usr/bin/python

## Script to evaluate quantisation gain settings using histograms
#  Assumes input file from observation script coldsky_gain_snapshots.py
# ./coldsky_gain_analysis.py --ant m062 --pol V --aux data/20151111-0001_progress.out /var/kat/archive/data/RTS/telescope_products/2015/11/11/1447232158.h5

from optparse import OptionParser
from datetime import datetime
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy, re, string, time
import os,sys
import katdal

if __name__ == '__main__':

    parser = OptionParser(usage='%prog [options] <path_to/filename.h5>', version="%prog 1.0")
    parser.add_option('--ant',
                      action='store',
                      dest='ant',
                      type=str,
                      default='m063',
                      help='Antenna to use, e.g.  \'m063\'.')
    parser.add_option('--pol',
                      action='store',
                      dest='pol',
                      type=str,
                      default='V',
                      help='Polarisation channel to use, e.g.  \'H or V\'.')
    parser.add_option('--aux',
                      action='store',
                      dest='aux',
                      type=str,
                      default=None,
                      help='Auxiliary files such as progress output for gain settings.')
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

    # Read data file
    try:
        h5 = katdal.open(datafile, quicklook=True)
    except Exception as err_msg: raise SystemExit('An error as occured:\n%s' % err_msg)

    # Read progress output
    if opts.aux is None: raise RuntimeError('Please provide progress.out file for gain settings')
    progress = opts.aux
    f = open(progress)
    myfile = f.readlines()
    f.close()

    # Correlate gain change timestamps from progress to spectra timestamps from observation file
    timestamps = []
    req_gains = []
    gain_idx = []
    for line in myfile:
        if line.strip().find("Set digital gain on selected DBE") > 0:
            _re_date = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d+.\d+', line.strip())
            timestamp = time.mktime(datetime.strptime(_re_date.group(), '%Y-%m-%d %H:%M:%S.%f').timetuple())
            timestamps.append(timestamp)
            gain_idx.append(numpy.argmin(abs((timestamp+2*3600)-numpy.array(h5.timestamps))))
            _re_gain = re.search('Set digital gain on selected DBE to (\d+)\.', line.strip())
            req_gains.append(_re_gain.group(1))
    req_gains=numpy.array(req_gains, dtype=int)

    # Channel indices for passband baseline
    passband = h5.channel_freqs
    min_idx=numpy.argmin(numpy.abs(passband-900e6))
    max_idx=numpy.argmin(numpy.abs(passband-1670e6))
    baseline_idx = range(min_idx, max_idx)
    print "Channel range %d to %d" % (min_idx, max_idx)
    print "Frequency range %f MHz to %f MHz" % (passband[min_idx]/1e6, passband[max_idx]/1e6)

    # Boltzman constant
    k = 1.38e-23
    Tsys = 35  # K
    BW   = h5.channel_freqs[-1]-h5.channel_freqs[0] + h5.channel_width # Hz
    Pnom = 10.*numpy.log10(k*Tsys*BW) + 30 # dBm
    print "RTS nominal input power %.2f [dBm]" % Pnom
    Ssys = 10.*numpy.log10(k*Tsys) + 30 # dBm/Hz
    print "Expected power spectral density of the system noise floor %.2d [dBm/Hz]" % Ssys

    # Filtered passband frequencies
    h5.select(reset='T')
    scan_indices = h5.scan_indices
    passband = h5.channel_freqs
    data = {opts.ant:{}}
    for pol in ['H', 'V']:
        h5.select(reset='T')
        h5.select(ants=opts.ant,pol=pol,corrprods='auto',scans='track')
        data[opts.ant][pol] = {'vis': numpy.abs(h5.vis[:])}

    # Spectra per gain setting
    requant_data={}
    for pol in data[opts.ant].keys():
        requant_data[pol]={}
        arb_nf = []
        for row in range(numpy.shape(data[opts.ant][pol]['vis'])[0]):
            spectrum = data[opts.ant][pol]['vis'][row,:,0]
            arb_nf.append(numpy.mean(10*numpy.log10(numpy.array(spectrum[baseline_idx])+1e-23))) #dB
        arb_nf = numpy.array(arb_nf)
        arb_nf[arb_nf==-numpy.inf]=0

        requant_data[pol]={'gain_nf':[], 'gain_spectrum':[]}
        for i in range(len(gain_idx)-1):
            requant_data[pol]['gain_nf'].append(numpy.median(arb_nf[gain_idx[i]:gain_idx[i+1]]))
            requant_data[pol]['gain_spectrum'].append(numpy.median(data[opts.ant][pol]['vis'][gain_idx[i]:gain_idx[i+1],:,0], axis=0))
        requant_data[pol]['gain_nf'].append(numpy.median(arb_nf[gain_idx[-1]:]))
        requant_data[pol]['gain_spectrum'].append(numpy.median(data[opts.ant][pol]['vis'][gain_idx[-1]:,:,0], axis=0))

    # Measure SNR
    eval_data={}
    for pol_idx in range(len(requant_data.keys())):
        pol = requant_data.keys()[pol_idx]
        valid_gain_idx = numpy.nonzero(numpy.array(requant_data[pol]['gain_nf'])>0)[0][0]
        noise_floor = numpy.array(requant_data[pol]['gain_nf'])[valid_gain_idx:-1]
        eval_data[pol]={'max_power':[], 'dc_power':[], 'snr':[]}
        for idx in range(valid_gain_idx,len(requant_data[pol]['gain_nf'])-1):
            spectrum = numpy.array(requant_data[pol]['gain_spectrum'])[idx,:]
            delta_P = requant_data[pol]['gain_nf'][idx] - Ssys
            eval_data[pol]['max_power'].append(numpy.max(10*numpy.log10(spectrum[baseline_idx])-delta_P))
            eval_data[pol]['snr'].append(numpy.max(10*numpy.log10(spectrum[baseline_idx])-delta_P)-Ssys)
            eval_data[pol]['dc_power'].append((10*numpy.log10(spectrum)-delta_P)[0])
    valid_gain_idx = numpy.nonzero(numpy.array(requant_data[opts.pol]['gain_nf'])>0)[0][0]
    inflect_idx = len(eval_data[opts.pol]['snr'])-numpy.argmax(eval_data[opts.pol]['snr'][::-1])-1
    use_idx = valid_gain_idx+inflect_idx

    # Generate output report
    pagetext = 'Pointing of a dish cold sky SCP\n'
    pagetext = pagetext + 'Look at cold sky with telescope at nominal input.\n'
    pagetext = pagetext + 'Set a gain value, capture autocorrelation spectra.\n'
    pagetext = pagetext + 'For each gain setting calculate the SNR as the power.\n'
    pagetext = pagetext + 'between the noisefloor and the maximum signal measured.\n'
    pagetext = pagetext + 'Find the inflection point where the SNR start to.\n'
    pagetext = pagetext + 'decrease with increase gain as optimal gain.\n'
    pagetext = pagetext + '\n'
    pagetext = pagetext + 'ReQuantisation gain for antenna %s using polarisation %s\n' % (opts.ant, opts.pol)
    pagetext = pagetext + 'Gain = %d\n' % (req_gains[use_idx])

    pp = PdfPages(opts.outfile+'.pdf')
    plt.figure()
    plt.axes(frame_on=False)
    plt.xticks([])
    plt.yticks([])
    plt.title("RTS Report %s"%opts.outfile,fontsize=14, fontweight="bold")
    plt.text(0,0,pagetext,fontsize=12)
    plt.savefig(pp,format='pdf')

    plt.figure(figsize=(15,9))
    plt.hold(True)
    for pol in data[opts.ant].keys():
        valid_gain_idx = numpy.nonzero(numpy.array(requant_data[pol]['gain_nf'])>0)[0][0]
        plt.plot(req_gains[valid_gain_idx:],numpy.array(requant_data[pol]['gain_nf'][valid_gain_idx:])+Ssys,
                 markersize=2,
                 label='Ant %s Pol %s' % (opts.ant, pol))
    plt.hold(False)
    plt.legend(loc=0)
    plt.xlabel('Requantisation gain')
    plt.ylabel('Avg power [arb dB]')
    plt.savefig(pp,format='pdf')

    plt.figure(figsize=(20,7))
    for pol_idx in range(len(requant_data.keys())):
        pol = requant_data.keys()[pol_idx]
        valid_gain_idx = numpy.nonzero(numpy.array(requant_data[pol]['gain_nf'])>0)[0][0]
        noise_floor = numpy.array(requant_data[pol]['gain_nf'])[valid_gain_idx:-1]
        inflect_idx = len(eval_data[pol]['snr'])-numpy.argmax(eval_data[pol]['snr'][::-1])-1
        plt.subplot(1,2,pol_idx)
        plt.hold(True)
        plt.plot(req_gains[valid_gain_idx:-1],eval_data[pol]['snr'], 'k.-', alpha=0.3, label='snr')
        plt.axvline(x=req_gains[valid_gain_idx+inflect_idx], color='y', linestyle='--')
        plt.hold(False)
        plt.legend(loc=0)
        plt.ylabel("Coarse measure of SNR [dB]")
        plt.xlabel('Requantisation gain [dB]')
        plt.title("Antenna %s Pol %s Gain %d" % (opts.ant,pol,req_gains[valid_gain_idx+inflect_idx]))
    plt.savefig(pp,format='pdf')


    gain_idx = req_gains[use_idx-1]
    plt.figure(figsize=(25,9))
    plt.hold(True)
    for pol in requant_data.keys():
        spectrum = numpy.array(requant_data[pol]['gain_spectrum'])[gain_idx,:]
        delta_P = requant_data[pol]['gain_nf'][gain_idx] - Ssys
        signal_pwr = numpy.max(10*numpy.log10(numpy.array(requant_data[pol]['gain_spectrum'])[gain_idx,:])-delta_P)
        spectrum_snr = signal_pwr - Ssys

        plt.plot(passband/1e6,
             10*numpy.log10(numpy.array(requant_data[pol]['gain_spectrum'])[gain_idx,:])-delta_P,
             alpha=0.5, label="Pol %s, Measured max power signal %.2f[dBm/Hz], SNR %.2f [dB]" % (pol, signal_pwr, spectrum_snr))
        plt.axhline(y=Ssys, color='c', alpha=0.5, linestyle='--')
        plt.axhline(y=signal_pwr, alpha=0.5, linestyle='--')
    plt.hold(False)
    plt.legend(loc=0)
    plt.xlabel('Freq [MHz]')
    plt.ylabel('Power [dBm/ch]')
    plt.title('Ant %s Gain %d' % (opts.ant, req_gains[gain_idx]))
    plt.savefig(pp,format='pdf')

    pp.close()

# -fin-

