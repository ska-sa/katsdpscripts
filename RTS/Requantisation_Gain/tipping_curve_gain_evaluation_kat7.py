#! /usr/bin/python

## Script to verify quantisation gain settings using commissioning tipping curve data

from optparse import OptionParser
import os,sys
import katfile
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
import logging
from collections import defaultdict
from katcp import BlockingClient, Message

class KatstoreClient(BlockingClient):

    def __init__(self, host, port, timeout = 15.0):
        super(KatstoreClient,self).__init__(host, port, timeout)
        self.timeout = timeout

    def __enter__(self):
        self.start(timeout=self.timeout)
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        self.join(timeout=self.timeout)

    def historical_sensor_data(self, sensor_names_filter, start_seconds, end_seconds,
                        period_seconds=-1, strategy='stepwise',
                        fetch_last_value=False, timeout=None):
        timeout = max(15, timeout or (end_seconds - start_seconds)/1000)
        reply, informs =  self.blocking_request(
            Message.request(
                'historical-sensor-data', sensor_names_filter,
                start_seconds, end_seconds, period_seconds,
                strategy, int(fetch_last_value), timeout),
            timeout = timeout
         )
        if reply.arguments[0] != 'ok' or int(reply.arguments[1]) == 0:
            return self._results(reply, None)
        else:
            return self._results(reply, informs)

    def _results(self, reply, informs):
        result_dict = defaultdict(list)
        if informs:
            for inform in informs:
                sensor_name, csv_data = inform.arguments
                data = result_dict[sensor_name]
                data.extend(csv_data.strip().split('\n'))
                result_dict[sensor_name] = data
        return result_dict

    def historical_sensor_list(self, sensor_filter=''):
        reply, informs = self.blocking_request(
                Message.request('historical-sensor-list', sensor_filter))
        if reply.arguments[0] == 'ok':
            result = [inform.arguments for inform in informs]
        else:
            logger.warn(reply)
            result = []
        return result

# Using the noise model associated with the input,
# calculate a polynomial fit for temperature calibration
def ND(noisemodel, passband):
    nd_freqs = numpy.array(noisemodel)[:,0]
    nd_temps = numpy.array(noisemodel)[:,1]
    passband_min_idx = numpy.argmin(numpy.abs(nd_freqs - passband[-1]))-1
    passband_max_idx = numpy.argmin(numpy.abs(nd_freqs - passband[0]))+1
    nd_freq_passband = nd_freqs[passband_min_idx:passband_max_idx]
    nd_temp_passband = nd_temps[passband_min_idx:passband_max_idx]
    coefficients  = numpy.polyfit(nd_freqs, nd_temps, 7)
    polynomial    = numpy.poly1d(coefficients)
    passband_tcal = numpy.array(polynomial(passband))
    return [nd_freqs, nd_temps, nd_freq_passband, nd_temp_passband, passband, passband_tcal]



if __name__ == '__main__':

    parser = OptionParser(usage='%prog [options] -f <filename.data> --ant <ant>', version="%prog 1.0")
    parser.add_option('-f', '--file',
                      action='store',
                      dest='datafile',
                      type=str,
                      default=None,
                      help='Full path to archive datafile, e.g. \
                      \'/var/kat/archive/data/comm/2014/03/04/1393935684.h5\'.')
    parser.add_option('--minchn',
                      action='store',
                      dest='minchn',
                      type=int,
                      default=180,
                      help='Min channel number defining bandpass, default = \'%default\'.')
    parser.add_option('--maxchn',
                      action='store',
                      dest='maxchn',
                      type=int,
                      default=800,
                      help='Max channel number defining bandpass, default = \'%default\'.')
    parser.add_option('--ant',
                      action='store',
                      dest='ant',
                      type=str,
                      default='all',
                      help='Antenna to use, e.g.  \'ant1\', default is to do analysis for all available antennas.')
    parser.add_option('-o', '--out',
                      action='store',
                      dest='outfile',
                      type=str,
                      default=None,
                      help='Full path name of output PDF report file.')
    (opts, args) = parser.parse_args()

    if opts.datafile is None or opts.ant is None: raise SystemExit(parser.print_usage())
    if opts.outfile is None: opts.outfile = os.path.splitext(os.path.basename(opts.datafile))[0]
    else:                    opts.outfile = os.path.splitext(os.path.basename(opts.outfile))[0]
    # Generate output report
    pagetext = 'Tipping curve data analysis to evaluate requantisation gain setting\n'
    pagetext += 'For each tipping curve,\n \
                 evaluate the height of the noise diode\n \
                 in relation to the noise floor per pointing\n'
    pagetext += 'For each linearity graph,\n \
                 evaluate linear relation between ADC and pointing power\n \
                 and spread of dump points per pointing should be tight\n'
    pp = PdfPages(opts.outfile+'.pdf')
    plt.figure()
    plt.axes(frame_on=False)
    plt.xticks([])
    plt.yticks([])
    plt.title("RTS Report %s"%opts.outfile,fontsize=14, fontweight="bold")
    plt.text(0,0,pagetext,fontsize=12)
    plt.savefig(pp,format='pdf')

##Read observation file
    try:
        h5 = katfile.open(opts.datafile, quicklook=True)
    except Exception as err_msg: raise SystemExit('An error as occured:\n%s' % err_msg)

    for pol in ['h','v']:
        print 'Analysis of tipping curve for %s' % (opts.ant+pol)
        inpt=opts.ant+pol
        h5.select(reset='T')
        h5.select(inputs=inpt, corrprods='auto', scans='track')
        scan_idx = h5.scan_indices
        passband = h5.channel_freqs[opts.minchn:opts.maxchn]
        channels = h5.channels[opts.minchn:opts.maxchn]

        if len(h5.vis[:].flatten())< 1:
            plt.figure()
            plt.axes(frame_on=False)
            plt.xticks([])
            plt.yticks([])
            pagetext = 'No data available for antenna %s' % opts.ant
            plt.text(0,0,pagetext,fontsize=12)
            plt.savefig(pp,format='pdf')
            pp.close()
            raise RuntimeError('No data available for input %s\n' %inpt)

##Display tipping curve data
        plt.figure(1)
        plt.clf()
        plt.subplots_adjust(hspace=.7)
        plt.subplots_adjust(wspace=.7)
        plt.subplot(2,1,1)
        plt.hold(True)
        plt.semilogy(h5.channel_freqs/1e6, numpy.median(numpy.abs(h5.vis[:]), axis=0), 'b')
        plt.semilogy(passband/1e6, numpy.median(numpy.abs(h5.vis[:, channels]), axis=0), 'g')
        plt.hold(False)
        plt.axis('tight')
        plt.xlabel('Feq [MHz]')
        plt.ylabel('Power [arb dB]')
        plt.title('Observation spectrum for input %s' %inpt)
        plt.subplot(2,1,2)
        plt.semilogy(numpy.median(numpy.abs(h5.vis[:]), axis=1), 'b')
        plt.axis('tight')
        plt.xlabel('Time [sec]')
        plt.ylabel('Power [arb dB]')
        plt.title('Tipping curve for input %s' %inpt)
        plt.savefig(pp,format='pdf')

# Use coupler noise diode mode to compute a model profile over the passband frequency range
        coupler_noise = h5.file['MetaData/Configuration/Antennas/%s/%s_coupler_noise_diode_model' % (opts.ant,pol)]
        [nd_freqs, nd_temps, nd_freq_passband, nd_temp_passband, noiseband, passband_tcal] = ND(coupler_noise, passband)
        plt.figure(2)
        plt.clf()
        plt.hold(True)
        plt.subplots_adjust(hspace=.7)
        plt.subplots_adjust(wspace=.7)
        plt.plot(nd_freqs/1e6, nd_temps, 'y',nd_freq_passband/1e6, nd_temp_passband, 'r')
        plt.plot(noiseband/1e6, passband_tcal, 'm:')
        plt.hold(False)
        plt.legend(['NS model', 'NS temp passband', 'Tcal passband'], 0)
        plt.ylabel('Temp [K]')
        plt.xlabel('Freq [MHz]')
        plt.title('Noise diode profile for input %s' %inpt)

# Calculate the temperature vs ADC power for each pointing in the tipping curve
        tipping_data = {'powers':{}}
        sensor_name = 'dbe7.dbe.%s.adc.power'%inpt
        tipping_data['powers']={'adc':[], 'pointing':[]}

        for idx in scan_idx:
            print 'Reading index %d of %d' % (idx, scan_idx[-1])
            h5.select(reset='T')
            h5.select(inputs=inpt, corrprods='auto', scans=idx)
            nd_sensor  = numpy.array(h5.sensor['Antennas/%s/nd_coupler'%(opts.ant)])
            vis_data   = numpy.mean(numpy.abs(h5.vis[:]),axis=1).flatten()
            timestamps = h5.timestamps[:]
            start = h5.timestamps[0]
            stop  = h5.timestamps[-1]


            # Calibrate power measurement assuming noise input signal
            S_on  = numpy.median(vis_data[nd_sensor][1:-1])
            S_off = numpy.median(vis_data[~nd_sensor])
            # calibration scale factor
            C = numpy.array(passband_tcal/numpy.abs(S_on-S_off))
            point_pwr = numpy.abs(h5.vis[:, channels])
            # Boltzman constant
            k = 1.38e-23
            # calibrated passband using Tsys noise diode calibration
            B_ch = (h5.spectral_windows[0]).channel_width
            cal_amp_W = k*point_pwr[:,:,0]*C*B_ch
            cal_amp = 10.*numpy.log10(cal_amp_W) + 30 # dBm
            tipping_data['powers']['pointing'].append(numpy.median(cal_amp[~nd_sensor], axis=1))

            # ADC input power from sensors
            with KatstoreClient(host = 'obs.kat7.karoo.kat.ac.za', port=2090) as katstore:
                data = []
                data = katstore.historical_sensor_data(sensor_name,
                                                       start_seconds=start,
                                                       end_seconds=stop)
                tipping_data['powers']['adc'].append(numpy.median(numpy.array([sensor.split(',')[-1] for sensor in data[sensor_name]],dtype=float)))

# Generate linearity graph
        plt.figure(3)
        plt.clf()
        plt.hold(True)
        median_pwr = []
        for idx in range(len(tipping_data['powers']['adc'])):
            pointing_powers = tipping_data['powers']['pointing'][idx]
            median_pwr.append(numpy.median(pointing_powers))
            adc_input_power = tipping_data['powers']['adc'][idx]
            plt.plot(numpy.ones(numpy.shape(pointing_powers))*adc_input_power, pointing_powers, 'r.')
        # fit linear relation
        z = numpy.polyfit(tipping_data['powers']['adc'],median_pwr,1)
        f = numpy.poly1d(z)
        plt.plot(tipping_data['powers']['adc'], f(tipping_data['powers']['adc']), 'y')
        plt.hold(False)
        plt.xlabel('ADC Input Power [dB]')
        plt.ylabel('Calibrated Power [dBm]')
        plt.title('Power linearity graph for input %s' %inpt)
        plt.savefig(pp,format='pdf')

    pp.close()
    print 'Done: Generated output report %s.pdf' %opts.outfile

# -fin-
