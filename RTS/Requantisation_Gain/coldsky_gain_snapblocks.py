#!/usr/bin/python
# Track target(s) and capture ADC and Quantiser snap shot output

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import numpy, time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint
# import time

# Capture snap block output
def capture_snap(kat, src_name, gain, filename):
#     nr_data_points = 16384*16 # largest buffer of ADC raw snap shot output

    # Create device array of antennas, based on specification string
    ants = kat.ants
    user_logger.info('Using antennas: %s' % (' '.join([ant.name for ant in ants]),))

    for ant in ants:
        user_logger.info('Capturing ADC snap blocks from antenna %s' %(ant.name))
        raw_adc_xdata = numpy.array([])
        raw_adc_ydata = numpy.array([])
	nr_data_points = 16384*4 # largest buffer of ADC raw snap shot output
        while len(raw_adc_xdata) < nr_data_points:
            # Capture raw adc output buffer on PPS signal (append samples to obtain signal of length 2^14)
#             raw_adc_xdata=numpy.hstack((raw_adc_xdata, numpy.array(ant.req.dig_adc_snap_shot(time.time(),'h').messages[1].arguments[1:], dtype=float)))
#             raw_adc_ydata=numpy.hstack((raw_adc_ydata, numpy.array(ant.req.dig_adc_snap_shot(time.time(),'v').messages[1].arguments[1:], dtype=float)))
	    # DBE digitiser snap block interface has changed 4 Aug 2015 (len=8192)
	    raw_adc_xdata=numpy.hstack((raw_adc_xdata, numpy.array([mesg.arguments[0] for mesg in ant.req.dig_adc_snap_shot('h',timeout=60).messages[1:]],dtype=float)))
# 	    time.sleep(1)
	    raw_adc_ydata=numpy.hstack((raw_adc_ydata, numpy.array([mesg.arguments[0] for mesg in ant.req.dig_adc_snap_shot('v',timeout=60).messages[1:]],dtype=float)))
# 	    time.sleep(1)

        user_logger.info('Capturing Quant snap blocks from antenna %s' %(ant.name))
        quant_adc_xdata = numpy.array([])
        quant_adc_ydata = numpy.array([])
 	nr_data_points = 16384*32 # larger number 32k has many more channels
        while len(quant_adc_xdata) < nr_data_points:
            # Capture quantiser output spectra
            xdata=kat.data_rts.req.cbf_quantiser_snapshot(ant.name+'h',timeout=60).messages[0].arguments[1]
# 	    time.sleep(1)
            quant_adc_xdata = numpy.hstack((quant_adc_xdata, numpy.array([complex(val.strip()) for val in xdata.split()])))
            ydata=kat.data_rts.req.cbf_quantiser_snapshot(ant.name+'v',timeout=60).messages[0].arguments[1]
# 	    time.sleep(1)
            quant_adc_ydata = numpy.hstack((quant_adc_ydata, numpy.array([complex(val.strip()) for val in ydata.split()])))

        # Write captured data to file
        try:
            fout = open(filename, 'a')
            fout.write('Source:\t %s \n' % src_name)
            fout.write('Gain:\t %d \n' % gain)
            fout.write('Antenna:\t %s \n' % ant.name)
            fout.write('ADC H:\t %s\n' % str(raw_adc_xdata.tolist())[1:-1])
            fout.write('ADC V:\t %s\n' % str(raw_adc_ydata.tolist())[1:-1])
            fout.write('Quant H:\t %s\n' % str(quant_adc_xdata.tolist())[1:-1])
            fout.write('Quant V:\t %s\n' % str(quant_adc_ydata.tolist())[1:-1])
            fout.close()
        except IOError:
            raise RuntimeError('Unable to open output file %s \n' % filename)

# code snippet stolen from Sean's dbe_gain_track.py script
def set_gains(kat,value):
    ants = kat.ants
    for ant in ants:
        for pol in ['h','v']:
            user_logger.info("Setting gain %d to antenna %s" % (int(value), '%s%s'%(ant.name,pol)))
            kat.data_rts.req.cbf_gain('%s%s'%(ant.name,pol), int(value))
	    time.sleep(1)


if __name__ == '__main__':

    # Set up standard script options
    usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
    description="Perform an imaging run of a specified target, capturing both ADC and Quantiser snapshots for each target"
    parser = standard_script_options(usage=usage,
                                     description=description)
    # Add experiment-specific options
    parser.add_option('-t', '--target-duration', type='float', default=30,
                      help='Minimum duration to track the imaging target per visit, in seconds (default="%default")')
    parser.add_option('--outfile', type='string', default='/home/kat/comm/ruby/quantisation/test_quantisation.data',
                      help='Outputfile containing snap shot data (default="%default")')
    parser.add_option('--step', type='int', default=1,
                      help='Integer increment size over gain range (default=%default)')
    parser.add_option('--min-gain', type='int', default=1,
                      help='Integer minimum requantisation gain setting (default=%default)')
    parser.add_option('--max-gain', type='int', default=300,
                      help='Integer maximum requantisation gain setting (default=%default)')

    # Set default value for any option (both standard and experiment-specific options)
    parser.set_defaults(description='Requantisation Gain Evaluation', nd_params='coupler,0,0,-1',dump_rate=0.1)
    # Parse the command line
    opts, args = parser.parse_args()

    # Check options and arguments, and build KAT configuration, connecting to proxies and devices
    if len(args) == 0:
        raise ValueError("Please specify the target(s) and calibrator(s) to observe as arguments, either as "
                         "description strings or catalogue filenames")
    with verify_and_connect(opts) as kat:
        sources = collect_targets(kat, args)
        user_logger.info("Imaging targets are [%s]" %
                         (', '.join([("'%s'" % (target.name,)) for target in sources]),))
        with start_session(kat, **vars(opts)) as session:
            # Start capture session, which creates HDF5 file
            session.standard_setup(**vars(opts))
            session.capture_start()

            for gain in range(opts.min_gain, opts.max_gain, opts.step):
                try:
                    if not opts.dry_run: set_gains(kat, int(gain))
                except Exception,  e: print e
                session.label('%s' % gain)
                user_logger.info("Set digital gain on selected DBE to %d." % gain)
                # Loop over sources in sequence
                for target in sources:
# RvR -- take out pointing instruction for snap shot debug
                    target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
                    user_logger.info("Initiating %g-second track on target '%s'" % (opts.target_duration, target.name))
                    # Set the default track duration for a target with no recognised tags
# RvR -- take out pointing instruction for snap shot debug
                    session.label('track')
                    session.track(target, duration=opts.target_duration)
                    # While still on source capture snap shots
                    if not kat.dry_run: capture_snap(kat, target.name, gain, opts.outfile)

# - fin -
