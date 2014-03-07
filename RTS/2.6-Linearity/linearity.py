#! /usr/bin/python

import katarchive
import katfile

import numpy
import pylab

## -- Globals --
DISPLAY = True
## -- Globals --

## -- Input --
infile = '1393418142.h5'
ant    = 'ant6'
pol    = 'h'
inpt   = ant + pol
## -- Input --

## -- Calculation of frequency to channel --
def freq2chan(freq_mhz, bw_mhz=400, n_chans=1024):
  chan_nr = round(float(freq_mhz)/float(bw_mhz)*n_chans)%n_chans
  return int(chan_nr)
## -- Calculation of frequency to channel --

## -- Power calibration --
def Tcal(vis, freq_idx, T):
  spectrum = numpy.mean(numpy.abs(vis), axis=0).flatten()
  # Tsys cal for frequency region
  track_values = numpy.mean(numpy.abs(vis[:,freq_idx]), axis=1)
  # the noise diode according to the observation script is always the first 10 dumps
  S_on  = numpy.mean(track_values[:10])
  S_off = numpy.mean(track_values[10:21])
  # calibration scale factor
  C = numpy.array(T/numpy.abs(S_on-S_off))
  # Power [dBm] on baseline region
  B_ch = (h5.spectral_windows[0]).channel_width # 400e6/1024
  k = 1.38e-23
  return (10*numpy.log10(k*spectrum*C*B_ch)+30)
## -- Power calibration --

## -- Input power from EIRP calculations --
EIRP = 60      # dBW
freq = 1.575e9 # Hz
l = (3e8/freq) # m
n=0.8
D=12           # m
d=38899e3      # m
gain = 10.*numpy.log10(n*(numpy.pi*D/l)**2) # dB
path_loss = 20.*numpy.log10(4*numpy.pi*d/l) # dB
## -- Input power from EIRP calculations --


## -- Main --
f = katarchive.get_archived_products(infile)
h5 = katfile.open(f[0])

# assume center frequency 1.575 GHz
passband = h5.channel_freqs
max_idx = numpy.argmin(numpy.abs(passband-1.5025e9))
min_idx = numpy.argmin(numpy.abs(passband-1.5075e9))
null_idx = range(min_idx, max_idx)
target_idx = range(370,630)    # channel indices of target
passband_idx = range(freq2chan(72),freq2chan(328))

# noise diode profile
sensor_data = h5.file['MetaData/Configuration/Antennas/%s/%s_coupler_noise_diode_model' % (ant,pol)]
nd_freqs = numpy.array(sensor_data)[:,0]
nd_temps = numpy.array(sensor_data)[:,1]
passband_min_idx = numpy.argmin(numpy.abs(nd_freqs - passband[passband_idx[-1]]))-1
passband_max_idx = numpy.argmin(numpy.abs(nd_freqs - passband[passband_idx[0]]))+1
nd_freq_passband = nd_freqs[passband_min_idx:passband_max_idx]
nd_temp_passband = nd_temps[passband_min_idx:passband_max_idx]
coefficients  = numpy.polyfit(nd_freqs, nd_temps, 7)
polynomial    = numpy.poly1d(coefficients)
Tcal_passband = numpy.array(polynomial(passband))
if DISPLAY:
  pylab.figure()
  pylab.clf()
  pylab.subplots_adjust(hspace=.7)
  pylab.subplots_adjust(wspace=.7)
  pylab.plot(nd_freqs/1e6, nd_temps, 'y',nd_freq_passband/1e6, nd_temp_passband, 'r')
  pylab.plot(passband/1e6, Tcal_passband, 'm:')
  pylab.legend(['NS model', 'NS temp passband', 'Tcal passband'], 0)
  pylab.ylabel('Temp [K]')
  pylab.xlabel('Freq [MHz]')
  pylab.title('Noise diode profile')


# Calibrate target spectrum
h5.select(reset='T')
h5.select(inputs=inpt,corrprods='auto',scans='track')
scan_indices = h5.scan_indices
# extract on target spectrum
h5.select(reset='T')
h5.select(inputs=inpt, corrprods='auto', scans=scan_indices[-1])
Ph_dbm = Tcal(h5.vis[:], target_idx, Tcal_passband)
if DISPLAY:
  pylab.figure()
  pylab.plot(passband[passband_idx]/1e6, Ph_dbm[passband_idx], 'b')
  pylab.axvline(x=passband[min_idx]/1e6, color='r')
  pylab.axvline(x=passband[max_idx]/1e6, color='r')
  pylab.legend(['GPS spectra', 'null win'],0)
  pylab.xlabel('Feq [MHz]')
  pylab.ylabel('Power [dBm]')
  pylab.title('GPS satellite, center freq = 1575 MHz')

# passband suppression round null frequency
target_offset = [5.000000, 4.897959, 4.795918, 4.693878, 4.591837, 4.489796, 4.387755, 4.285714, 4.183673, 4.081633, 3.979592, 3.877551, 3.775510, 3.673469, 3.571429, 3.469388, 3.367347, 3.265306, 3.163265, 3.061224, 2.959184, 2.857143, 2.755102, 2.653061, 2.551020, 2.448980, 2.346939, 2.244898, 2.142857, 2.040816, 1.938776, 1.836735, 1.734694, 1.632653, 1.530612, 1.428571, 1.326531, 1.224490, 1.122449, 1.020408, 0.918367, 0.816327, 0.714286, 0.612245, 0.510204, 0.408163, 0.306122, 0.204082, 0.102041, 0.000000]
compression = []
for idx in range(1,len(scan_indices)):
  h5.select(reset='T')
  h5.select(inputs=inpt, corrprods='auto', scans=scan_indices[idx])
  Ph_dbm = Tcal(h5.vis[:], target_idx, Tcal_passband)
  compression.append(numpy.mean(Ph_dbm[null_idx]))

P1dB = numpy.mean(compression[1:10])-1
cmp_idx = numpy.argmin(numpy.abs(compression-P1dB))

if DISPLAY:
  pylab.figure()
  pylab.plot(target_offset[1:],compression[1:])
  pylab.axhline(y=P1dB, color='r')
  pylab.gca().invert_xaxis()
  pylab.legend(['compression','P1dB'],0)
  pylab.ylabel('Power [dBm]')
  pylab.xlabel('Offset [deg]')
  pylab.title('Baseline compression at %f [deg]' % target_offset[cmp_idx])

h5.select(reset='T')
h5.select(inputs=inpt, corrprods='auto', scans=scan_indices[cmp_idx+1])
Ph_dbm = Tcal(h5.vis[:], target_idx, Tcal_passband)
# pylab.figure()
# pylab.plot(passband[passband_idx]/1e6, Ph_dbm[passband_idx], 'b')
# pylab.axvline(x=passband[min_idx]/1e6, color='r')
# pylab.axvline(x=passband[max_idx]/1e6, color='r')
# pylab.legend(['GPS spectra', 'null win'],0)
# pylab.xlabel('Feq [MHz]')
# pylab.ylabel('Power [dBm]')
# pylab.title('GPS satellite offset angle %f [deg]' % target_offset[cmp_idx])

# Input power needed to produce 1dB compression
loss = numpy.mean(compression[1:10]) -1 - compression[cmp_idx] # dB
print 'Input power to produce 1dB compression = %f [dBm]' % (numpy.max(Ph_dbm[target_idx]) + gain - loss)
print "Receiver power %f [dBm]" % (EIRP - path_loss + gain)

pylab.show()


# -fin-

