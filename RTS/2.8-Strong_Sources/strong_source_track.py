#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger


def read_sensors(ants, sensorlist):
    sensors = {}
    for ant in ants.clients:
        for sen in sensorlist:
            clean_sen = sen.replace('.', '_').replace('-', '_')
            sensors["%s_%s" % (ant.name, clean_sen)] = ant.sensor.get(clean_sen).get_value()
    return sensors


def compare_sensors(sensors1, sensors2, num):
    """ return True if sensors2 - sensors1 > num"""
    return_value = False
    for sen in sensors1.keys():
        if sensors2[sen] - sensors1[sen] > num:
            return_value = True
            user_logger.error('%s has changed by %g from %g to %g')
    return return_value


# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'>  [--cold-target=<'target/catalogue'> ...]",
                                 description='Track 2 sources, one strong source which is bracketed by the cold sky source'
                                             'for a specified time. The strong target must be specified.'
                                             'The first valid source in the catalogue give will be used.'
                                             'The script will terminate with an error state if the LNA'
                                             'temperature rises by 15 degrees from the start.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=7200.0,
                  help='Length of time to track the main source in seconds (default=%default)')
parser.add_option('--cold-duration', type='float', default=900.0,
                  help='Length of time to track the cold sky source in seconds when bracketing the observation (default=%default)')
parser.add_option('--cold-target', type='string', default="SCP,radec,0,-90",
                  help='The target/catalogue of the cold sky source to use when bracketing the observation (default=%default)')
parser.add_option('--sensor-watch',  default="rsc.rxl.omt.temperature,rsc.rxl.cryostat.body-temperature",
                  help='comma delimited list of sensors to monitor while the observation is in progress (default=%default)')
parser.add_option('--change-limit', type='float', default=15.0,
                  help='The amount of +change in the sensor that is allowed to happen bet the script exits.(default=%default)')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Strong Sources track', dump_rate=0.1, nd_params='coupler,20,0,20')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")


sensorlist = opts.sensor_watch.split(',')
endobs = False

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    strong_sources = collect_targets(kat, args)
    cold_sources = collect_targets(kat, [opts.cold_target])
    sensors = read_sensors(kat.ants, sensorlist)

    # Get band info from subarray controller
    band = kat.sub.sensor.band.get_value()
    if not band:
        raise ValueError("Please ensure that the subarray has a well-defined band, not %r" % (band,))
    # Get existing attenuation settings
    attenuation_old = {}
    atten_sensor = {}
    for ant in kat.ants:
        for pol in 'hv':
            input_name = ant.name + pol
            atten_name = "dig_{}_band_rfcu_{}pol_attenuation".format(band, pol)
            atten_sensor[input_name] = ant.sensor[atten_name]  # Deal with KeyError if no such sensor
            attenuation_old[input_name] = atten_sensor[input_name].get_value()
            user_logger.info("%s %s pol band %r has attenuation = %f",
                             ant.name, pol, band, attenuation_old[input_name])

    # Quit early if there are no sources to observe
    valid_targets = True
    if not strong_sources.filter(el_limit_deg=opts.horizon):
        user_logger.warning("No strong source targets are currently visible - please re-run the script later")
        valid_targets = False
    if not cold_sources.filter(el_limit_deg=opts.horizon):
        user_logger.warning("No cold source targets are currently visible - please re-run the script later")
        valid_targets = False
    if valid_targets:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            session.standard_setup(**vars(opts))
            session.capture_start()
            target_list = []
            target_list.append((cold_sources, opts.cold_duration, "track_before", -1))  # -1 means to use the old attenuation
            target_list.append((strong_sources, opts.track_duration, "track_strong", 39))
            target_list.append((cold_sources, opts.cold_duration, "track_after", -1))  # -1 means to use the old attenuation
            for observation_sources, track_duration, label, attenuation in target_list:
                for ant in kat.ants:
                    for pol in 'hv':
                        input_name = ant.name + pol
                        if attenuation == -1:
                            attenuation = attenuation_old[input_name]
                        ant.req.dig_attenuation(pol, attenuation, timeout=30)
                        user_logger.info("%s %s pol, attenuation set to %f",
                                         ant.name, pol, atten_sensor[input_name].get_value())
                # Iterate through source list, picking the first one that is up
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                    session.label(label)
                    user_logger.info("Initiating %g-second track on target '%s'" % (track_duration, target.name,))
                    # Split the total track on one target into segments lasting as long as the noise diode period
                    # This ensures the maximum number of noise diode firings
                    start_time = time.time()
                    while (time.time() - start_time) < track_duration:
                        next_track = track_duration - (time.time() - start_time)
                        # Cut the track short if time ran out
                        if opts.nd_params['period'] > 0:
                            next_track = min(next_track, opts.nd_params['period'])
                        if next_track <= 0 or not session.track(target, duration=next_track, announce=False):
                            user_logger.info("Exiting Time Loop")
                            break
                        if compare_sensors(sensors, read_sensors(kat.ants, sensorlist), opts.change_limit):
                            user_logger.error("Sensor Compare Failed. Ending obsevation ")
                            endobs = True
                            break
                    if endobs: break
                if endobs: break
    # Restore attenuation settings
    for ant in kat.ants:
        for pol in 'hv':
            input_name = ant.name + pol
            ant.req.dig_attenuation(pol, attenuation_old[input_name], timeout=30)
            user_logger.info("%s %s pol, attenuation set to %f",
                             ant.name, pol, atten_sensor[input_name].get_value())
