
#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
#import katpoint
import scpi as SCPI
 
# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which script will end '
                       'as soon as the current track finishes (no limit by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum duration (which must be set for this)')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')

parser.add_option('--siggen-ip',  default='192.168.14.61',
                  help='Signal Generator IP adress (default=%default)')
parser.add_option('--siggen-port', type='int', default=5025,
                  help='Signal Generator port (default=%default)')
parser.add_option('--siggen-freq', type='float', default=600.0,
                  help='Signal Generator frequency  in MHz(default=%default)')
parser.add_option('--siggen-power', type='float', default=-30.0,
                  help='Signal Generator power in dBm (default=%default)')
parser.add_option('--force-siggen', action="store_true", default=False,
                  help='Force the Signal Generator commands during dry run as a test')


# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='UHF signal generator track',dump_rate=1.0)
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    user_logger.info("Default SCP source added to catalogue")
    args.append('SCP,radec,0,-90')

if  opts.siggen_power > -20.:
    raise ValueError("Please specify a Signal Generator power less than -20 dBm")


siggen_ip = opts.siggen_ip
siggen_port = opts.siggen_port
siggen_freq = opts.siggen_freq
siggen_power = opts.siggen_power
# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    if opts.force_siggen and  kat.dry_run: user_logger.warning("The signal generator commands are being used during a dry-run")
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        user_logger.error("Unable to set Antenna mode to 'STOP'.")

    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            if not opts.no_delays and not kat.dry_run :
                if session.dbe.req.auto_delay('on'):
                    user_logger.info("Turning on delay tracking.")
                else:
                    user_logger.error('Unable to turn on delay tracking.')
            elif opts.no_delays and not kat.dry_run:
                if session.dbe.req.auto_delay('off'):
                    user_logger.info("Turning off delay tracking.")
                else:
                    user_logger.error('Unable to turn off delay tracking.')
                if session.dbe.req.zero_delay():
                    user_logger.info("Zeroed the delay values.")
                else:
                    user_logger.error('Unable to zero delay values.')

            user_logger.info("Setting up the signal Generator ip:port %s:%i."%(siggen_ip,siggen_port))
            if not kat.dry_run or opts.force_siggen : # prevent verifiing script from messing with things and failing to connect
                sig=SCPI(siggen_ip,siggen_port)
                testcon = sig.testConnect()
                if testcon == False:
                    user_logger.error('Test connection to signal generator failed.')
                else:
                    user_logger.info("Connected to Signal Generator:%s"%(testcon))
                    sig.reset()
                    user_logger.info("Signal Generator reset")
                    sig.outputOn()
                    user_logger.info("Signal Generator output on")
                    sig.setFrequency(siggen_freq*1.0e6)
                    user_logger.info("Signal Generator frequency is set to %7.3f MHz"%(sig.getFrequency()*1.0e-6 ))
                    siggen_freq = sig.getFrequency()
                    sig.setPower(siggen_power)
                    user_logger.info("Signal Generator Power is set to %f dBm"%(sig.getPower()))
                    siggen_power=sig.getPower()
                    

        
## Using SCPI class for comms to signal generator for CW input signal

            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            while keep_going:
                keep_going = (opts.max_duration is not None) and opts.repeat
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that is up
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                    session.label('siggen,f=%f,p=%f,'%(siggen_freq,siggen_power ))
                    user_logger.info("Initiating %g-second track on target '%s'" % (opts.track_duration, target.name,))
                    # Split the total track on one target into segments lasting as long as the noise diode period
                    # This ensures the maximum number of noise diode firings
                    total_track_time = 0.
                    while total_track_time < opts.track_duration:
                        next_track = opts.track_duration - total_track_time
                        # Cut the track short if time ran out
                        if opts.max_duration is not None:
                            next_track = min(next_track, opts.max_duration - (time.time() - start_time))
                        if opts.nd_params['period'] > 0:
                            next_track = min(next_track, opts.nd_params['period'])
                        if next_track <= 0 or not session.track(target, duration=next_track, announce=False):
                            break
                        total_track_time += next_track
                    if opts.max_duration is not None and (time.time() - start_time >= opts.max_duration):
                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script" %
                                            (opts.max_duration,))
                        keep_going = False
                        break
                    targets_observed.append(target.name)
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
            if not kat.dry_run or opts.force_siggen : # prevent verifiing script from messing with things and failing to connect
                user_logger.info("Turning Off Signal Generator RF Power")
                sig.outputOff()
                user_logger.info("Closing connection to Signal Generator")
                sig.__close__()
            