from __future__ import with_statement

from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger
import time
import numpy as np
import katpoint

targets = {'ant1' : (25.119, -8.944, 0.083),
           'ant2' : (90.315, 26.648, -0.067),
           'ant3' : (3.989, 26.925, -0.006),
           'ant4' : (-21.600, 25.500, 0.000),
#           'ant1' : (18.4, -8.7, 0),
#           'ant2' : (86.2, 25.5, 0),
#           'ant3' : (3.2, 27.3, 0),
#           'ant4' : (-21.6, 25.5, 0),
           'ant5' : (-37.5, -1.3, 0),
           'ant6' : (-61.5, -78.0, 0),
           'ant7' : (-87.8, 76.3, 0),
           'asc' : (57, -27, 0),
           '12m' : (45, -43, 0),
#           'asc' : (46, -27, 0),
#           '12m' : (33, -43, 0),
           'minister' : (40., -40., 0),
           'origin' : (63.7, -32.9, 0)}


def track(ants,targets,duration=10):
    # send this target to the antenna.
    for target,ant_x in zip(targets,ants):
        ant_x.req.target(target)
        ant_x.req.mode("POINT")
        user_logger.info("Slewing %s to target : %s"%(ant_x.name,target,))
    # wait for antennas to lock onto target
    locks = 0
    for ant_x in ants:
        if ant_x.wait("lock", True, 300): locks += 1
    if len(ants) == locks:
        user_logger.info("Tracking Target : %s for %s seconds"%(target,str(duration)))
        time.sleep(duration)
        user_logger.info("Target tracked : %s "%(target,))
        return True
    else:
        user_logger.warning("Unable to track Targe : %s "%(target,))
        return False


def enu_to_azel(e, n, u):
    """Convert vector in ENU coordinates to (az, el) spherical coordinates.

    This converts a vector in the local east-north-up (ENU) coordinate system to
    the corresponding horizontal spherical coordinates (azimuth and elevation
    angle). The ENU coordinates can be in any unit, as the vector length will be
    normalised in the conversion process.

    Parameters
    ----------
    e, n, u : float or array
        East, North, Up coordinates (any unit)

    Returns
    -------
    az_rad, el_rad : float or array
        Azimuth and elevation angle, in radians

    """
    return np.arctan2(e, n), np.arctan2(u, np.sqrt(e * e + n * n))

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="%prog [options] <target>",
                               description="Point dishes at the given target and record data.")
# Generic options
parser.add_option('-m', '--max-duration', dest='max_duration', type="float", default=60.0,
                  help='Duration to run experiment, in seconds (default=%default)')
parser.set_defaults(description='Point to enu')
(opts, args) = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify one target argument (via name or coords, e.g. 'ant1' or '(50,50,0)')")
elif len(args) > 1:
    raise ValueError("Please specify only one target argument (if using coords, don't include spaces, e.g. use '(50,50,0)')")

target = args[0]

if target in targets:
    target_enu = targets[target]
else:
    try:
        target_enu = tuple(float(coord) for coord in target.strip('\n\t ()[]').split(','))
        target = target.replace(',', '/')
    except ValueError:
        raise ValueError("Unknown target '%s', should be one of %s" % (target, targets.keys()))
if len(target_enu) != 3:
    raise ValueError("Please provide 3 coordinates (east, north, up)")


# Various non-optional options...
if opts.description is 'point to enu':
    opts.description = "Data recorded while pointing at '%s'" % target

with verify_and_connect(opts) as kat:
    kat.ants.req.sensor_sampling("lock","event")
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        session.label('track')
        session.ants.req.drive_strategy('shortest-slew')
        session.ants.req.sensor_sampling("lock","event")
        target_list = []
        for ant in session.ants:
            antenna = katpoint.Antenna(ant.sensor.observer.get_value())
            enu = np.asarray(target_enu) - np.asarray(antenna.position_enu)
            if np.all(enu == 0):
                enu = np.array([0, 0, 1])
            az, el = enu_to_azel(*enu)
            az, el = katpoint.rad2deg(az), katpoint.rad2deg(el)
            # Go to nearest point on horizon if target is below elevation limit
            el = max(el, 3.0)
            target_description = "%s, azel, %f, %f" % (target, az, el)
            target_list.append(target_description)
        track(session.ants,target_list,duration=opts.max_duration)
   
