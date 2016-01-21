#! /usr/bin/python

from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
from scipy import integrate

import katdal

import matplotlib
import numpy
import os
import pylab

## GLOBAL SCRIPT CONSTANTS ##
# Boltzman constant
k = 1.38e-23
# Linearity spec
spec = -57  # dBm

# GPS satellite (SatelliteCalculator.xls)
dfnull    = 10.23e6      # Hz
dfpeak    = 3.*dfnull/2. # Hz
nullidx   = 8

# Filtered passband frequencies
f_start = 900e6        # Hz
f_stop  = 1670e6       # Hz

# Channel indices over target
dft=20e6               # Hz
# Channel indices for null
dfn=10e6               # Hz

# Default noise diode model files
default_path='/home/kat/svn/katconfig/user/noise-diode-models/mkat'
## GLOBAL SCRIPT CONSTANTS ##

##Read input H5 file
def readOBS(filename, ant, pol, fc, fnull):
    try:
        h5 = katdal.open(filename, quicklook=True)
    except Exception as err_msg: raise SystemExit('An error has occured:\n%s' % err_msg)
    h5.select(reset='T')
    h5.select(ants=ant,pol=pol,corrprods='auto',scans='track')
    visibilities = h5.vis[:]
    scan_indices = h5.scan_indices
    passband = h5.channel_freqs
    chan_bw = (h5.spectral_windows[0]).channel_width
    the_pointings = []
    for idx in scan_indices:
        h5.select(reset='T')
        h5.select(ants=ant,pol=pol,corrprods='auto',scans=idx)
        the_pointings.append(h5.shape[0])

    # Channel indices for passband baseline
    min_idx=numpy.argmin(numpy.abs(passband-f_start))
    max_idx=numpy.argmin(numpy.abs(passband-f_stop))
    baseline_idx = range(min_idx, max_idx)

    # Extract passband only
    visibilities = visibilities[:,baseline_idx,:]
    passband = passband[baseline_idx]
    # Channel indices over target
    min_idx = numpy.argmin(numpy.abs(passband-(fc-dft/2.)))
    max_idx = numpy.argmin(numpy.abs(passband-(fc+dft/2.)))
    target_range = range(min_idx, max_idx)
    # channel indices for null
    min_idx = numpy.argmin(numpy.abs(passband-(fnull-dfn/2)))
    max_idx = numpy.argmin(numpy.abs(passband-(fnull+dfn/2)))
    null_range = range(min_idx, max_idx)

    return [the_pointings, visibilities, chan_bw, passband, target_range, null_range]

##Noise diode profile over passband frequency range
def noiseDIODE(passband, noisemodel=None, tsys=-1.):
    if noisemodel is not None:
        # Read model data [freq Hz, Te K]
        nd_freqs = numpy.array(noisemodel)[:,0]
        nd_temps = numpy.array(noisemodel)[:,1]
    else:
        nd_freqs = passband
        nd_temps = tsys*numpy.ones(numpy.shape(passband))

    coefficients  = numpy.polyfit(nd_freqs, nd_temps, 1)
    polynomial    = numpy.poly1d(coefficients)
    tcal_passband = numpy.array(polynomial(passband))

    return [nd_freqs, nd_temps, tcal_passband]

##Calibrate observation data
def calibTsys(the_pointings, visibilities, chan_bw, target_offset, tcal_range, null_range, target_range):
    Pgps= []
    Pns = []
    # Per pointing calibration and analysis
    cntr = 0
    for point in range(len(the_pointings)):
        nr = the_pointings[point]
        vis = visibilities[cntr+1:cntr+nr,:,:]
        cntr += nr

        # Calculate calibration factor for pointing from noise diode over null frequency range
        null_vis = numpy.mean(numpy.abs(vis[:,null_range]), axis=1)
        # find noise diode on and off samples
        threshold = numpy.average(null_vis)
        # everything above the threshold = with noise diode
        src_nd_idx = numpy.nonzero(null_vis > threshold)[0][1:-1]
        if len(src_nd_idx)<3:
            target_offset = numpy.delete(target_offset, point)
            continue
        S_on = numpy.mean(null_vis[src_nd_idx])
        # everything below the threshold = without noise diode
        src_idx = numpy.nonzero(null_vis < threshold)[0][1:-1]
        if len(src_idx)<3:
            target_offset = numpy.delete(target_offset, point)
            continue
        S_off = numpy.mean(null_vis[src_idx])
        # calibration scale factor
        C = numpy.array(tcal_range/numpy.abs(S_on-S_off))

        # Apply calibration factor to spectrum data
        mean_amp = numpy.mean(numpy.abs(vis), axis=0).flatten()
        cal_amp_W = k*mean_amp*chan_bw*C
        cal_amp_dBm = 10.*numpy.log10(cal_amp_W) + 30

        # Compute the target and noise floor total power using numerical intergration over the frequency ranges
        Pgps.append(10.*numpy.log10(integrate.simps(cal_amp_W[target_range],passband[target_range])) + 30)
        Pns.append(10.*numpy.log10(numpy.average(cal_amp_W[null_range])) + 30)

    return [Pgps, Pns]

## -- Main --
if __name__ == '__main__':

    usage = "python %prog [options] <filename> \
\n\t Where filename is the full path name of H5 observation file' \
\nExample: \
\n\t python %prog --ant m063 --pol H --tsys 20 --offset 7 --nsteps 50 1432552011.h5 \
"
    parser = OptionParser(usage=usage, version="%prog 1.0")
# Test parameters
    parser.add_option('--ant',
                      action='store',
                      dest='ant',
                      type=str,
                      default=None,
                      help='Antenna to use, e.g. \'m063\'.')
    parser.add_option('--pol',
                      action='store',
                      dest='pol',
                      type=str,
                      default=None,
                      help='Polarisation, horizontal (\'H\') or vertical (\'V\').')
    parser.add_option('--nsf',
                      action='store',
                      dest='noisefile',
                      type=str,
                      default=None,
                      help='File containing noise diode model.')
    parser.add_option('--tsys',
                      action='store',
                      dest='tsys',
                      type=float,
                      default=-1.,
                      help='Use a constant Tsys for noise model.')
# Observation parameters from progress output
    parser.add_option('--offset',
                      action='store',
                      dest='offset',
                      type=float,
                      default=-1.,
                      help='Maximum offset angle from GPS.')
    parser.add_option('--nsteps',
                      action='store',
                      dest='nsteps',
                      type=float,
                      default=-1.,
                      help='Nr of offset steps onto GPS.')
    parser.add_option('--gps',
                      action='store',
                      dest='gps',
                      type=float,
                      default=1227.6e6,
                      help='Frequency of expected target in Hz, default = \'%default\' Hz.')
# Output and stuff
    parser.add_option('--out',
                      action='store',
                      dest='outfile',
                      type=str,
                      default=None,
                      help='Name of output report file.')
    parser.add_option('-v', '--verbose',
                      action='store_true',
                      dest='verbose',
                      default=False,
                      help='Display cross matched coordinates')

    (opts, args) = parser.parse_args()

##Test required input parameters
    if len(args) < 1:
        parser.print_usage()
        raise SystemExit('Observation data file is a required parameter.')

    if opts.ant is None:
        parser.print_usage()
        raise SystemExit('Antenna name is a required parameter.')
    if opts.pol is None:
        parser.print_usage()
        raise SystemExit('Antenna polarisation channel is a required parameter.')

    if opts.offset < 0 or opts.nsteps < 0:
        parser.print_usage()
        raise SystemExit('Observational parameters unknown, provide offset angle and number steps.')

    # Function takes in a single file -- specified as input argument -- for processing
    if len(args) > 1:
        parser.print_usage()
        raise SystemExit('Multiple input filenames given, only single input file expected')
    filename = args[0]
    if opts.noisefile is None and opts.tsys < 0:
        # use default name (bad choice)
        import string
        opts.noisefile = os.path.join(default_path,'rx.l.4.%s.csv'%string.lower(opts.pol))

##Observation test parameters
    fc     = opts.gps             # Hz
    fnull  = (fc-nullidx*dfnull)  # Hz
    delta      =opts.offset/opts.nsteps
    target_offset = numpy.arange(opts.offset,0-delta,-delta)

    outfile = opts.outfile
    if outfile is None: outfile = os.path.splitext(os.path.basename(filename))[0]
    outfile = outfile + '_' + opts.ant + '_' + opts.pol + '_linearity'

##Read input H5 file
    [the_pointings, visibilities, chan_bw, passband, target_range, null_range] = readOBS(filename, ant=opts.ant, pol=opts.pol, fc=fc, fnull=fnull)
##Noise diode profile or Tsys values given
    if opts.noisefile is not None:
        ##Read data from file
        fin = open(opts.noisefile, 'r')
        # Read and ignore header line
        fin.readline()
        fin.readline()
        # Read noise model data
        noisemodel=[]
        for line in fin.readlines():
            try:
                noisemodel.append(numpy.array(line.strip().split(','), dtype=float))
            except: print line.strip()
        fin.close()
        [nd_freqs, nd_temps, tcal] = noiseDIODE(passband, noisemodel=noisemodel)
    else:
        [nd_freqs, nd_temps, tcal] = noiseDIODE(passband, tsys=opts.tsys)

    [Pgps, Pns] = calibTsys(the_pointings, visibilities, chan_bw, target_offset, tcal, null_range, target_range)
    # ignore suspect pointings at the beginning
    Pgps=Pgps[3:] 
    Pns =Pns[3:]
    target_offset=target_offset[3:]

    # find inflection point using GPS power calculated
    inf_idx = numpy.argmax(numpy.abs(numpy.diff(Pgps)))
    # gps_offset_nonlinear = target_offset[inf_idx+1]
    gps_offset_nonlinear = target_offset[inf_idx]
    offset_idx = numpy.argmin(numpy.abs(gps_offset_nonlinear-target_offset))

    # approximate dynamic range reading
    visibilities = numpy.array(visibilities)
    noisefloor = numpy.mean(10.*numpy.log10(visibilities.mean(axis=0)))
    maxsignal = numpy.max(10.*numpy.log10(visibilities.max(axis=0)))
    dr = (maxsignal-noisefloor)

##Test results
    # Generate output report
    h5 = katdal.open(filename, quicklook=True)
    with PdfPages(outfile+'.pdf') as pdf:
        pagetext  = "\nLinearity requirement: (P1dB input >= %d dBm)" % spec
        pagetext += "\n\nDescription: %s\nName: %s\nExperiment ID: %s" %(h5.description, h5.name, h5.experiment_id)
        pagetext += "\n\nAntenna: %s\nPolarisation: %s" %(opts.ant, opts.pol)
        pagetext  = pagetext + "\n"
        pagetext += "\n\nTest Setup:"
        pagetext += "\nPassband frequency range %.2f MHz to %.2f MHz" % (passband[0]/1e6, passband[-1]/1e6)
        pagetext += "\nGPS target range %.2f MHz to %.2f MHz around fc=%.2f MHz" % (passband[target_range[0]]/1e6, passband[target_range[-1]]/1e6, fc/1e6)
        pagetext += "\nGPS null range %.2f MHz to %.2f MHz around fnull=%.2f MHz" % (passband[null_range[0]]/1e6, passband[null_range[-1]]/1e6, fnull/1e6)
        pagetext += "\nScan offsets from max %.2f deg onto target using %d steps" % (opts.offset, opts.nsteps)
        pagetext += "\n\nMeasurement results:"
        pagetext += '\nCompression inflection at %.2f [deg] offset with GPS signal power %.2f [dBm]' % \
                   (gps_offset_nonlinear, Pgps[inf_idx])
                   # (gps_offset_nonlinear, Pgps[offset_idx])
        pagetext += '\nAverage power %.2f [dBm] at %.2f [MHz] null' % (numpy.mean(Pns[:inf_idx]), fnull/1e6)
        pagetext += "\n\nInput power spec\n"
        # if (Pgps[offset_idx]-spec) < 0: pagetext += '\n\n[Fail] Receiver power at receiver %.2f dB < %.2f dBm' % (abs(Pgps[offset_idx]-spec), spec)
        if (Pgps[offset_idx]-spec) < 0: pagetext += '\n\n[Fail] Receiver power at receiver %.2f dB < %.2f dBm' % (abs(Pgps[inf_idx]-spec), spec)
        # else: pagetext +='\n\n[Success] Receiver power at compression angle %.2f dB >= %.2f dBm' % (abs(Pgps[offset_idx]-spec), spec)
        else: pagetext +='\n[Success] Receiver power at compression angle %.2f dB >= %.2f dBm' % (abs(Pgps[inf_idx]-spec), spec)
        pagetext += "\n\nDynamic range\n"
        if (dr - 27) < 0: pagetext += '\n\n[Fail] Not enough dynamic range %.2f dB < 27 dB' % numpy.abs(dr)
        else: pagetext += '\n[Success] Enough dynamic range %.2f dB >= 27 dB' % numpy.abs(dr)

        pylab.figure(None,figsize = (16,8))
        pylab.axes(frame_on=False)
        pylab.xticks([])
        pylab.yticks([])
        pylab.title("RTS Report %s"%outfile,fontsize=14, fontweight="bold")
        pylab.text(0,0,pagetext,fontsize=12)
        pdf.savefig()
        pylab.close()

        pylab.figure(None,figsize = (16,8))
        pylab.subplots_adjust(hspace=.3)
        pylab.subplot(211)
        pylab.hold(True)
        pylab.axvline(x=passband[null_range[0]]/1e6, color='g')
        pylab.axvline(x=passband[target_range[0]]/1e6, color='r')
        pylab.axvline(x=passband[null_range[-1]]/1e6, color='g')
        pylab.axvline(x=passband[target_range[-1]]/1e6, color='r')
        pylab.semilogy(passband/1e6, numpy.mean(numpy.abs(visibilities), axis=0), 'b')
        pylab.hold(False)
        pylab.axis('tight')
        pylab.legend(['null win', 'target'],0)
        pylab.xlabel('Feq [MHz]')
        pylab.ylabel('Power [arb dB]')
        pylab.title('GPS satellite, center freq = %.2f MHz' % (fc/1e6))
        pylab.subplot(212)
        pylab.hold(True)
        track_means = numpy.mean(numpy.abs(visibilities), axis=1)
        pointings = numpy.arange(numpy.sum(the_pointings))
        cntr=0
        S_on = []
        S_off = []
        for idx in range(len(the_pointings)):
            nr = the_pointings[idx]
            threshold = numpy.average(track_means[cntr+1:cntr+nr])
            # everything above the threshold = with noise diode
            src_nd_idx = numpy.nonzero(track_means[cntr+1:cntr+nr] > threshold)[0][1:-1]
            S_on.append(numpy.mean(track_means[cntr+1:cntr+nr][src_nd_idx]))
            # everything below the threshold = without noise diode
            src_idx = numpy.nonzero(track_means[cntr+1:cntr+nr] < threshold)[0][1:-1]
            S_off.append(numpy.mean(track_means[cntr+1:cntr+nr][src_idx]))
            pylab.plot(pointings[cntr+1:cntr+nr], track_means[cntr+1:cntr+nr], '.-')
            pylab.plot(pointings[cntr+1:cntr+nr][src_nd_idx], track_means[cntr+1:cntr+nr][src_nd_idx], 'r.')
            pylab.plot(pointings[cntr+1:cntr+nr][src_idx], track_means[cntr+1:cntr+nr][src_idx], 'b.')
            cntr += nr
        pylab.plot(pointings[:], track_means[:], 'r:')
        pylab.hold(False)
        pylab.title('Ant %s, Pol %s' % (opts.ant, opts.pol))
        pylab.axis('tight')
        pylab.xlabel('Pointings [#]')
        pdf.savefig()

        pylab.figure(None,figsize = (16,8))
        pylab.clf()
        pylab.hold(True)
        pylab.plot(nd_freqs/1e6, nd_temps, 'y')
        pylab.plot(passband/1e6, tcal, 'm:')
        pylab.hold(False)
        pylab.legend(['NS model', 'NS temp passband', 'Tcal passband'], 0)
        pylab.ylabel('Temp [K]')
        pylab.xlabel('Freq [MHz]')
        pylab.title('Noise diode profile')
        pdf.savefig()

        pylab.figure(None,figsize = (16,8))
        # for GPS and other CW RFI sources this inflexion point will be obvious
        pylab.plot(target_offset[:len(Pgps)][:-1],numpy.diff(Pgps), 'b.-')
        pylab.axvline(x=target_offset[:len(Pgps)][inf_idx], color='r', linestyle=':')
        pylab.gca().invert_xaxis()
        # pylab.title('Inflection at %.2f [deg] offset, signal power %.2f [dBm]' % (gps_offset_nonlinear, Pgps[inf_idx+1]), fontsize=12)
        pylab.title('Inflection at %.2f [deg] offset, signal power %.2f [dBm]' % (gps_offset_nonlinear, Pgps[inf_idx]), fontsize=12)
        pylab.ylabel('Gain [dB]', fontsize=12)
        pylab.xlabel('Lower Offset Angle [deg]', fontsize=12)
        pdf.savefig()

        d = pdf.infodict()
        import datetime
        d['Title'] = h5.description
        d['Author'] = 'Ruby van Rooyen'
        d['Subject'] = 'RTS linearity system engineering spec test'
        d['CreationDate'] = datetime.datetime(2015, 06, 30)
        d['ModDate'] = datetime.datetime.today()

    print "Test report %s.pdf generated" % outfile

    if opts.verbose:
      pylab.show()

    # cleanup before exit
    try: pylab.close('all')
    except: pass # nothing to close

# -fin-
