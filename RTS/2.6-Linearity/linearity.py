#! /usr/bin/python

from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
from scipy import integrate

import katdal

import matplotlib
import numpy
import os
import pylab


## -- Noise diode profile over passband frequency range
def NoiseProfile(noise_model, frequency_range):
  nd_freqs = numpy.array(noise_model)[:,0]
  nd_temps = numpy.array(noise_model)[:,1]
  min_idx = numpy.argmin(numpy.abs(nd_freqs - frequency_range[-1]))-1
  max_idx = numpy.argmin(numpy.abs(nd_freqs - frequency_range[0]))+1
  nd_freq_range = nd_freqs[min_idx:max_idx]
  nd_temp_range = nd_temps[min_idx:max_idx]
  coefficients  = numpy.polyfit(nd_freqs, nd_temps, 7)
  polynomial    = numpy.poly1d(coefficients)
  Tcal_range    = numpy.array(polynomial(frequency_range))

  pylab.figure()
  pylab.clf()
  pylab.hold(True)
  pylab.subplots_adjust(hspace=.7)
  pylab.subplots_adjust(wspace=.7)
  pylab.plot(nd_freqs/1e6, nd_temps, 'y',nd_freq_range/1e6, nd_temp_range, 'r')
  pylab.plot(frequency_range/1e6, Tcal_range, 'm:')
  pylab.hold(False)
  pylab.legend(['NS model', 'NS temp passband', 'Tcal passband'], 0)
  pylab.ylabel('Temp [K]')
  pylab.xlabel('Freq [MHz]')
  pylab.title('Noise diode profile')

  return Tcal_range
## -- Noise diode profile over passband frequency range

## -- Tsys calibration: C = Tcal/(Son-Soff)
def Tcal(vis, freq_idx, noise_model):
  spectrum = numpy.mean(numpy.abs(vis), axis=0).flatten()
  # Tsys cal for frequency region
  track_means = numpy.mean(numpy.abs(vis[:,freq_idx]), axis=1)
  # the noise diode according to the observation script is always the first 10 dumps
#   S_on  = numpy.mean(track_means[:10])
#   S_off = numpy.mean(track_means[10:21])
  threshold = numpy.average(track_means)
# everything above the threshold = with noise diode
  src_nd_idx = numpy.nonzero(track_means > threshold)[0]
  S_on = numpy.mean(track_means[src_nd_idx])
  # everything below the threshold = without noise diode
  src_idx = numpy.nonzero(track_means < threshold)[0]
  S_off = numpy.mean(track_means[src_idx])
  # calibration scale factor
  C = numpy.array(noise_model/numpy.abs(S_on-S_off))
  return [spectrum, C]
## -- Tsys calibration: C = Tcal/(Son-Soff)

## -- 1dB compression point and headroom
def Headroom(power, target_offset):
  # indices over range where the system is moving on source, but is linear
  lin_off = target_offset[-15:-10]
  lin_gps = power[-15:-10]
  p = numpy.polyfit(lin_off,lin_gps,1)
  # fit a curve over the target region
  coefficients = numpy.polyfit(target_offset[-15:], power[-15:], 2)
  polynomial = numpy.poly1d(coefficients)
  nx = numpy.arange(target_offset[-1], target_offset[-15], 0.001)[::-1]
  line = p[0]*nx+p[1]
  curve = polynomial(nx)
  zero_crossings = numpy.where(numpy.diff(numpy.sign((line-curve)-1)))[0]
  pylab.figure()
  pylab.hold(True)
  pylab.plot(target_offset, power, 'y.:')
  pylab.plot(lin_off, lin_gps, 'b:')
  pylab.plot(nx, curve, 'm-')
  pylab.plot(nx, line, 'r-')
  pylab.axvline(x=nx[zero_crossings[-1]], color='g', linestyle=':')
  pylab.axhline(y=curve[zero_crossings[-1]], color='g')
  pylab.hold(False)
  pylab.gca().invert_xaxis()
  pylab.ylabel('Headroom [dB]')
  pylab.xlabel('Offset [deg]')
  pylab.title('Headroom to P1dB = %d dB' % (curve[zero_crossings[-1]]))

  return curve[zero_crossings[-1]]
## -- 1dB compression point and headroom

## -- Generate output report --
def Report(pp, h5, data):
  pagetext = "Description: %s\nName: %s\nExperiment ID: %s\n\n" %(h5.description, h5.name, h5.experiment_id)
  pagetext = pagetext + "Antenna: %s\nPolarisation: %s\n" %(data['ant'], data['pol'])
  pagetext = pagetext + "\n"
  pagetext = pagetext + "Measurement results\n"
  pagetext = pagetext + 'Headroom to 1dB compression point = %.2f [dB]\n' % data['headroom']
  pylab.figure()
  pylab.axes(frame_on=False)
  pylab.xticks([])
  pylab.yticks([])
  pylab.title("RTS Report %s"%outfile,fontsize=14, fontweight="bold")
  pylab.text(0,0,pagetext,fontsize=12)
  pylab.savefig(pp,format='pdf')
  pylab.close()

  figures=[manager.canvas.figure
           for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
  for i, figure in enumerate(figures):
    figure.savefig(pp,format='pdf')
## -- Generate output report --

## -- Use headroom as a measure of linearity
def Linearity(h5, ant, pol, null_hz, target_hz):
  inpt = ant + pol

  # identify target spectrum observations
  h5.select(reset='T')
  h5.select(inputs=inpt,corrprods='auto',scans='track')
  scan_indices = h5.scan_indices
  passband     = h5.channel_freqs
  nr_channels  = h5.channels
  channel_bw   = (h5.spectral_windows[0]).channel_width # Hz
  bandwidth    = channel_bw*len(nr_channels) # Hz

  # channel indices for null -- range of 10 MHz
  max_idx = numpy.argmin(numpy.abs(passband-(null_hz-5e6/2)))
  min_idx = numpy.argmin(numpy.abs(passband-(null_hz+5e6/2)))
  null_range = range(min_idx, max_idx)

  # channel indices over target -- range of 40 MHz
  max_idx = numpy.argmin(numpy.abs(passband-(target_hz-20e6/2)))
  min_idx = numpy.argmin(numpy.abs(passband-(target_hz+20e6/2)))
  target_range = range(min_idx, max_idx)

  pylab.figure()
  pylab.semilogy(passband[1:]/1e6, numpy.mean(numpy.abs(h5.vis[:]), axis=0)[1:], 'b')
  pylab.axvline(x=passband[null_range[0]]/1e6, color='g')
  pylab.axvline(x=passband[target_range[0]]/1e6, color='r')
  pylab.axvline(x=passband[null_range[-1]]/1e6, color='g')
  pylab.axvline(x=passband[target_range[-1]]/1e6, color='r')
  pylab.axis('tight')
  pylab.legend(['GPS spectra', 'null win', 'target'],0)
  pylab.xlabel('Feq [MHz]')
  pylab.ylabel('Power [dBm]')
  pylab.title('GPS satellite, center freq = %.2f MHz' % (target_hz/1e6))

  # noise diode profile
  noise_model   = h5.file['MetaData/Configuration/Antennas/%s/%s_coupler_noise_diode_model' % (ant,pol)]
  Tcal_passband = NoiseProfile(noise_model, passband)

  # calibrate measured temperatures
  k = 1.38e-23
  Pns = []
  Pgps = []
  for idx in range(2,len(scan_indices)):
    h5.select(reset='T')
    h5.select(inputs=inpt, corrprods='auto', scans=scan_indices[idx])
    [spectrum, Tcal_factor] = Tcal(h5.vis[:], null_range, Tcal_passband)
    # apply calibration and compute integrated power over noise floor (null region) and target
    calib_vis=k*numpy.array(Tcal_factor)*numpy.array(spectrum)
    Pns.append(10.*numpy.log10(numpy.average(calib_vis[null_range])*bandwidth))
    Pgps.append(10.*numpy.log10(integrate.simps((calib_vis[target_range]-numpy.average(calib_vis[null_range])).flatten(),passband[target_range][::-1])*channel_bw))

  # approximate off target degrees from observation output
  target_offset = numpy.arange(5,-0.1, -0.1)[::-1][:len(scan_indices)-1][::-1]
  # identify 1dB compression point and read off headroom to 1dB compression point
  cal_pwr = numpy.array(Pgps) - numpy.array(Pns)
  return Headroom(cal_pwr, target_offset[1:])
## -- Use headroom as a measure of linearity


## -- Main --
if __name__ == '__main__':

  usage = "\npython %prog [options] -f <filename>"
  parser = OptionParser(usage=usage, version="%prog 1.0")
  parser.add_option('-f', '--file',
                    action='store',
                    dest='filename',
                    type=str,
                    default=None,
                    help='Full path name of H5 observation file, e.g. \'/var/kat/archive/data/comm/2014/02/26/1393418142.h5\'.')
  parser.add_option('--ant',
                    action='store',
                    dest='ant',
                    type=str,
                    default='all',
                    help='Antenna to use, e.g. \'ant1\', default is to do analysis for all available antennas.')
  parser.add_option('--pol',
                    action='store',
                    dest='pol',
                    type=str,
                    default='all',
                    help='Polarisation, horisontal (\'h\') or vertical (\'v\'), default is to do analysis for both polarisations.')
  parser.add_option('--null',
                    action='store',
                    dest='null',
                    type=float,
                    default=1503.81e6,
                    help='Frequency of expected null in Hz, default = \'%default\' Hz.')
  parser.add_option('--target',
                    action='store',
                    dest='target',
                    type=float,
                    default=1575e6,
                    help='Frequency of expected target in Hz, default = \'%default\' Hz.')
  parser.add_option('--out',
                    action='store',
                    dest='outfile',
                    type=str,
                    default=None,
                    help='Name of output report file.')

  (opts, args) = parser.parse_args()

  if opts.filename is None: raise SystemExit(parser.print_usage())

  try:
    h5 = katdal.open(opts.filename, quicklook=True)
  except Exception as err_msg: raise SystemExit('An error as occured:\n%s' % err_msg)

  outfile = opts.outfile
  if outfile is None: outfile = os.path.splitext(os.path.basename(opts.filename))[0]

  ants = [opts.ant]
  pols = [opts.pol]
  if opts.ant == 'all':
    ants = [ant.name for ant in h5.ants]
    outants = 'all_antennas'
  else: outants = opts.ant
  if opts.pol == 'all':
    pols = ['h', 'v']
    outpols = 'H_V'
  else: outpols = opts.pol
  outfile = outfile + '_' + outants + '_' + outpols + '_linearity'

  # Generate output report
  pp = PdfPages(outfile+'.pdf')

  for ant in ants:
    for pol in pols:
      print 'Headroom analysis for antenna %s polarisation %s' % (ant, pol)
      headroom_1db = Linearity(h5, ant, pol, opts.null, opts.target)
      Report(pp, h5, {'headroom':headroom_1db, 'ant':ant, 'pol':pol})
      try: pylab.close('all')
      except: pass # nothing to close

  # cleanup before exit
  pp.close()
  try: pylab.close('all')
  except: pass # nothing to close

# -fin-
