"""Set of useful routines to do standard observations with KAT."""

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
import logging
import sys
import optparse
import uuid

import numpy as np
import katpoint
# This is used to document available spherical projections (and set them in case of TimeSession)
from katcore.proxy.antenna_proxy import AntennaProxyModel, Offset

from .array import Array
from .katcp_client import KATDevice
from .defaults import user_logger
from .utility import tbuild
from .misc import dynamic_doc

# Obtain list of spherical projections and the default projection from antenna proxy
_projections, _default_proj = AntennaProxyModel.PROJECTIONS, AntennaProxyModel.DEFAULT_PROJECTION
# Move default projection to front of list
_projections.remove(_default_proj)
_projections.insert(0, _default_proj)

def ant_array(kat, ants, name='ants'):
    """Create sub-array of antennas from flexible specification.

    Parameters
    ----------
    kat : :class:`utility.KATHost` object
        KAT connection object
    ants : :class:`Array` or :class:`KATDevice` object, or list, or string
        Antennas specified by an Array object containing antenna devices, or
        a single antenna device or a list of antenna devices, or a string of
        comma-separated antenna names, or the string 'all' for all antennas
        controlled via the KAT connection associated with this session

    Returns
    -------
    array : :class:`Array` object
        Array object containing selected antenna devices

    Raises
    ------
    ValueError
        If antenna with a specified name is not found on KAT connection object

    """
    if isinstance(ants, Array):
        return ants
    elif isinstance(ants, KATDevice):
        return Array(name, [ants])
    elif isinstance(ants, basestring):
        if ants.strip() == 'all':
            return kat.ants
        else:
            try:
                return Array(name, [getattr(kat, ant.strip()) for ant in ants.split(',')])
            except AttributeError:
                raise ValueError("Antenna '%s' not found (i.e. no kat.%s exists)" % (ant, ant))
    else:
        # The default assumes that *ants* is a list of antenna devices
        return Array(name, ants)

class ScriptLogHandler(logging.Handler):
    """Logging handler that writes logging records to HDF5 file via k7writer.

    Parameters
    ----------
    kat : :class:`utility.KATHost` object
        KAT connection object associated with this experiment

    """
    def __init__(self, kat):
        logging.Handler.__init__(self)
        self.kat = kat

    def emit(self, record):
        """Emit a logging record."""
        try:
            msg = self.format(record)
            self.kat.dbe.req.k7w_script_log(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class CaptureInitError(Exception):
    """Failure to start new capture session."""
    pass

class CaptureSession(object):
    """Context manager that encapsulates a single data capturing session.

    A data capturing *session* results in a single data file, potentially
    containing multiple scans and compound scans. An *experiment* may consist of
    multiple sessions. This object ensures that the capturing process is
    started and completed cleanly, even if exceptions occur during the session.
    It also provides canned routines for simple observations such as tracks,
    single scans and raster scans on a specific source.

    The initialisation of the session object selects a sub-array of antennas and
    does basic preparation of the data capturing subsystem (k7writer), which
    also opens the HDF5 output file. The setup is usually completed by calling
    :meth:`standard_setup` on the instantiated session object.

    The antenna specification *ants* do not have a default, which forces the
    user to specify them explicitly. This is for safety reasons, to remind
    the user of which antennas will be moved around by the script. The
    *observer* and *description* similarly have no default, to force the
    user to document the observation to some extent.

    Parameters
    ----------
    kat : :class:`utility.KATHost` object
        KAT connection object associated with this experiment
    experiment_id : string
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc.
    observer : string
        Name of person doing the observation
    description : string
        Short description of the purpose of the capturing session
    ants : :class:`Array` or :class:`KATDevice` object, or list, or string
        Antennas that will participate in the capturing session, as an Array
        object containing antenna devices, or a single antenna device or a
        list of antenna devices, or a string of comma-separated antenna
        names, or the string 'all' for all antennas controlled via the
        KAT connection associated with this session
    stow_when_done : {True, False}, optional
        If True, stow the antennas when the capture session completes.
    kwargs : dict, optional
        Ignore any other keyword arguments (simplifies passing options as dict)

    Raises
    ------
    ValueError
        If antenna with a specified name is not found on KAT connection object

    """
    def __init__(self, kat, experiment_id, observer, description, ants, stow_when_done=False, **kwargs):
        try:
            self.kat = kat
            self.experiment_id = experiment_id
            self.ants = ants = ant_array(kat, ants)
            ant_names = [ant.name for ant in ants.devs]
            self.stow_when_done = stow_when_done
            # By default, no noise diodes are fired
            self.nd_params = {'diode' : 'coupler', 'on' : 0., 'off' : 0., 'period' : -1.}
            self.last_nd_firing = 0.

            # Prepare the capturing system, which opens the HDF5 file
            reply = kat.dbe.req.k7w_capture_init()
            if not reply.succeeded:
                raise CaptureInitError(reply[1])
            # Enable logging to the new HDF5 file via the usual logger
            self._script_log_handler = ScriptLogHandler(kat)
            user_logger.addHandler(self._script_log_handler)

            user_logger.info('==========================')
            user_logger.info('New data capturing session')
            user_logger.info('--------------------------')
            user_logger.info('Experiment ID = %s' % (experiment_id,))
            user_logger.info('Observer = %s' % (observer,))
            user_logger.info("Description ='%s'" % description)
            user_logger.info('Antennas used = %s' % (' '.join(ant_names),))
            # Obtain the name of the file currently being written to
            reply = kat.dbe.req.k7w_get_current_file()
            outfile = reply[1] if reply.succeeded else '<unknown file>'
            user_logger.info('Opened output file %s' % (outfile,))
            user_logger.info('')

            # Log details of the script to the back-end
            kat.dbe.req.k7w_set_script_param('script-starttime', time.time())
            kat.dbe.req.k7w_set_script_param('script-name', sys.argv[0])
            kat.dbe.req.k7w_set_script_param('script-arguments', ' '.join(sys.argv[1:]))
            kat.dbe.req.k7w_set_script_param('script-experiment-id', experiment_id)
            kat.dbe.req.k7w_set_script_param('script-observer', observer)
            kat.dbe.req.k7w_set_script_param('script-description', description)
            kat.dbe.req.k7w_set_script_param('script-ants', ','.join(ant_names))
        except Exception, e:
            user_logger.error('CaptureSession failed to initialise (%s)' % (e,))
            raise

    def __enter__(self):
        """Enter the data capturing session."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the data capturing session, closing the data file."""
        if exc_value is not None:
            user_logger.error('Session interrupted by exception (%s)' % (exc_value,))
        self.end()
        # Do not suppress any exceptions that occurred in the body of with-statement
        return False

    def standard_setup(self, centre_freq=1800.0, dump_rate=1.0,
                       nd_params={'diode' : 'coupler', 'on' : 10.0, 'off' : 10.0, 'period' : 180.}, **kwargs):
        """Set up LO frequency, dump rate and noise diode parameters.

        This performs basic setup of the LO frequency, dump rate and noise diode
        parameters. It also sets strategies on antenna sensors that might be
        waited on. It should usually be called as the first step in a new session
        (unless the experiment has special requirements, such as holography).

        Parameters
        ----------
        centre_freq : float, optional
            RF centre frequency, in MHz
        dump_rate : float, optional
            Correlator dump rate, in Hz
        nd_params : dict, optional
            Dictionary containing parameters that control firing of the noise
            diode. These parameters are in the form of keyword-value pairs, and
            matches the parameters of the :meth:`fire_noise_diode` method.
        kwargs : dict, optional
            Ignore any other keyword arguments (simplifies passing options as dict)

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        session.nd_params = nd_params

        user_logger.info('RF centre frequency = %g MHz, dump rate = %g Hz' % (centre_freq, dump_rate))
        if nd_params['period'] > 0:
            user_logger.info("Will switch '%s' noise diode on for %g s and off for %g s, every %g s if possible" %
                             (nd_params['diode'], nd_params['on'], nd_params['off'], nd_params['period']))
        elif nd_params['period'] == 0:
            user_logger.info("Will switch '%s' noise diode on for %g s and off for %g s at every opportunity" %
                             (nd_params['diode'], nd_params['on'], nd_params['off']))
        else:
            user_logger.info("Noise diode will not fire automatically")
        # Log parameters to output file
        kat.dbe.req.k7w_set_script_param('script-rf-params',
                                         'Centre freq=%g MHz, Dump rate=%g Hz' % (centre_freq, dump_rate))
        kat.dbe.req.k7w_set_script_param('script-nd-params', 'Diode=%s, On=%g s, Off=%g s, Period=%g s' %
                                         (nd_params['diode'], nd_params['on'],
                                          nd_params['off'], nd_params['period']))

        # Setup strategies for the sensors we might be wait()ing on
        ants.req.sensor_sampling('lock', 'event')
        ants.req.sensor_sampling('scan.status', 'event')
        ants.req.sensor_sampling('mode', 'event')

        # Set centre frequency in RFE stage 7
        kat.rfe7.req.rfe7_lo1_frequency(4200.0 + centre_freq, 'MHz')
        effective_lo_freq = (centre_freq - 200.0) * 1e6
        # The DBE proxy needs to know the dump period (in ms) as well as the effective LO freq,
        # which is used for fringe stopping (eventually). This sets the delay model and other
        # correlator parameters, such as the dump rate, and instructs the correlator to pass
        # its data to the k7writer daemon (set via configuration)
        kat.dbe.req.capture_setup(1000.0 / dump_rate, effective_lo_freq)

        # If the DBE is simulated, it will have position update commands
        if hasattr(kat.dbe.req, 'dbe_pointing_az') and hasattr(kat.dbe.req, 'dbe_pointing_el'):
            first_ant = ants.devs[0]
            # The minimum time between position updates is fraction of dump period to ensure fresh data at every dump
            update_period_sec = 0.4 / dump_rate
            # Tell the position sensors to report their values periodically at this rate
            # Remember that this should be an *integer* number of milliseconds
            first_ant.sensor.pos_actual_scan_azim.set_strategy('period', str(int(1000 * update_period_sec)))
            first_ant.sensor.pos_actual_scan_elev.set_strategy('period', str(int(1000 * update_period_sec)))
            # Tell the DBE simulator where the first antenna is so that it can generate target flux at the right time
            first_ant.sensor.pos_actual_scan_azim.register_listener(kat.dbe.req.dbe_pointing_az, update_period_sec)
            first_ant.sensor.pos_actual_scan_elev.register_listener(kat.dbe.req.dbe_pointing_el, update_period_sec)
            user_logger.info("DBE simulator receives position updates from antenna '%s'" % (first_ant.name,))
        user_logger.info("--------------------------")

    def capture_start(self):
        """Start capturing data to HDF5 file."""
        # This starts the SPEAD stream on the DBE
        self.kat.dbe.req.dbe_capture_start()

    def label(self, label):
        """Add timestamped label to HDF5 file."""
        self.kat.dbe.req.k7w_set_label(label)

    def on_target(self, target):
        """Determine whether antennas are tracking a given target.

        If all connected antennas in the sub-array participating in the session
        have the given *target* as target and are locked in mode 'POINT', we
        conclude that the array is on target.

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target to check, as an object or description string

        Returns
        -------
        on_target : {True, False}
            True if antennas are tracking the given target

        """
        # Turn target object into description string (or use string as is)
        target = getattr(target, 'description', target)
        for ant in self.ants.devs:
            if not ant.is_connected():
                continue
            if (ant.sensor.target.get_value() != target) or (ant.sensor.mode.get_value() != 'POINT') or \
               (ant.sensor.lock.get_value() != '1'):
                return False
        return True

    def target_visible(self, target, duration=0., timeout=300., horizon=2.):
        """Check whether target is visible for given duration.

        This checks whether the *target* is currently above the given *horizon*
        and also above the horizon for the next *duration* seconds, taking into
        account the *timeout* on slewing to the target. If the target is not
        visible, an appropriate message is logged. The target location is not
        very accurate, as it does not include refraction, and this is therefore
        intended as a rough check only.

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target to check, as an object or description string
        duration : float, optional
            Duration of observation of target, in seconds
        timeout : float, optional
            Timeout involved when antenna cannot reach the target
        horizon : float, optional
            Elevation limit serving as horizon, in degrees

        Returns
        -------
        visible : {True, False}
            True if target is visible from all antennas for entire duration

        """
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        horizon = katpoint.deg2rad(horizon)
        # Include an average time to slew to the target (worst case about 90 seconds, so half that)
        now = time.time() + 45.
        average_el, visible_before, visible_after = [], [], []
        for ant in self.ants.devs:
            if not ant.is_connected():
                continue
            antenna = katpoint.Antenna(ant.sensor.observer.get_value())
            az, el = target.azel(now, antenna)
            average_el.append(katpoint.rad2deg(el))
            # If not up yet, see if the target will pop out before the timeout
            if el < horizon:
                now += timeout
                az, el = target.azel(now, antenna)
            visible_before.append(el >= horizon)
            # Check what happens at end of observation
            az, el = target.azel(now + duration, antenna)
            visible_after.append(el >= horizon)
        if all(visible_before) and all(visible_after):
            return True
        always_invisible = any(~np.array(visible_before) & ~np.array(visible_after))
        if always_invisible:
            user_logger.warning("Target '%s' is never up during requested period (average elevation is %g degrees)" %
                                (target.name, np.mean(average_el)))
        else:
            user_logger.warning("Target '%s' will rise or set during requested period" % (target.name,))
        return False

    def fire_noise_diode(self, diode='coupler', on=10.0, off=10.0, period=0.0, announce=True):
        """Switch noise diode on and off.

        This switches the selected noise diode on and off for all the antennas
        doing the observation.

        The on and off durations can be specified. Additionally, setting the
        *period* allows the noise diode to be fired on a semi-regular basis. The
        diode will only be fired if more than *period* seconds have elapsed since
        the last firing. If *period* is 0, the diode is fired unconditionally.
        On the other hand, if *period* is negative it is not fired at all.

        Parameters
        ----------
        diode : {'coupler', 'pin'}
            Noise diode source to use (pin diode is situated in feed horn and
            produces high-level signal, while coupler diode couples into
            electronics after the feed at a much lower level)
        on : float, optional
            Minimum duration for which diode is switched on, in seconds
        off : float, optional
            Minimum duration for which diode is switched off, in seconds
        period : float, optional
            Minimum time between noise diode firings, in seconds. (The maximum
            time is determined by the duration of individual slews and scans,
            which are considered atomic and won't be interrupted.) If 0, fire
            diode unconditionally. If negative, don't fire diode at all.
        announce : {True, False}, optional
            True if start of action should be announced, with details of settings

        Returns
        -------
        fired : {True, False}
            True if noise diode fired

        Notes
        -----
        When the function returns, data will still be recorded to the HDF5 file.
        The specified *off* duration is therefore a minimum value. Remember to
        run :meth:`end` to close the file and finally stop the observation
        (automatically done when this object is used in a with-statement)!

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # If period is non-negative, quit if it is not yet time to fire the noise diode
        if period < 0.0 or (time.time() - session.last_nd_firing) < period:
            return False
        # Find pedestal controllers with the same number as antennas (i.e. 'ant1' maps to 'ped1') and put into Array
        pedestals = Array('peds', [getattr(kat, 'ped' + ant.name[3:]) for ant in ants.devs])

        if announce:
            user_logger.info("Firing '%s' noise diode (%g seconds on, %g seconds off)" % (diode, on, off))
        else:
            user_logger.info('firing noise diode')

        # Switch noise diode on on all antennas
        pedestals.req.rfe3_rfe15_noise_source_on(diode, 1, 'now', 0)
        # If using DBE simulator, fire the simulated noise diode for desired period to toggle power levels in output
        if hasattr(kat.dbe.req, 'dbe_fire_nd'):
            kat.dbe.req.dbe_fire_nd(on)
        time.sleep(on)
        # Mark on -> off transition as last firing
        session.last_nd_firing = time.time()
        # Switch noise diode off on all antennas
        pedestals.req.rfe3_rfe15_noise_source_on(diode, 0, 'now', 0)
        time.sleep(off)
        user_logger.info('noise diode fired')

        return True

    def set_target(self, target):
        """Set target to use for tracking or scanning.

        This sets the target on all antennas involved in the session, as well as
        on the DBE (where it serves as delay-tracking centre). It also moves the
        test target in the DBE simulator to match the requested target (if it is
        a stationary 'azel' type).

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target as an object or description string

        """
        # Create reference to KAT object and antennas, as this allows easy copy-and-pasting from this function
        kat, ants = self.kat, self.ants
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)

        # Set the antenna target (antennas will already move there if in mode 'POINT')
        ants.req.target(target)
        # Provide target to the DBE proxy, which will use it as delay-tracking center
        kat.dbe.req.target(target)
        # If using DBE simulator and target is azel type, move test target here (allows changes in correlation power)
        if hasattr(kat.dbe.req, 'dbe_test_target') and target.body_type == 'azel':
            azel = katpoint.rad2deg(np.array(target.azel()))
            kat.dbe.req.dbe_test_target(azel[0], azel[1], 100.)

    def track(self, target, duration=20.0, drive_strategy='longest-track', label='track', announce=True):
        """Track a target.

        This tracks the specified target with all antennas involved in the
        session.

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target to track, as an object or description string
        duration : float, optional
            Minimum duration of track, in seconds
        drive_strategy : {'longest-track', 'shortest-slew'}, optional
            Drive strategy employed by antennas, used to decide what to do when
            target is in azimuth overlap region of antenna. The default is to
            go to the wrap that will permit the longest possible track before
            the target sets.
        label : string, optional
            Label associated with compound scan in HDF5 file, usually single word.
        announce : {True, False}, optional
            True if start of action should be announced, with details of settings

        Returns
        -------
        success : {True, False}
            True if track was successfully completed

        Notes
        -----
        When the function returns, the antennas will still track the target and
        data will still be recorded to the HDF5 file. The specified *duration*
        is therefore a minimum value. Remember to run :meth:`end` to close the
        file and finally stop the observation (automatically done when this
        object is used in a with-statement)!

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)

        if announce:
            user_logger.info("Initiating %g-second track on target '%s'" % (duration, target.name))
        if not session.target_visible(target, duration):
            user_logger.warning("Skipping track, as target '%s' will be below horizon" % (target.name,))
            return False

        # Set the drive strategy for how antenna moves between targets, and the target
        ants.req.drive_strategy(drive_strategy)
        session.set_target(target)
        session.label(label)

        session.fire_noise_diode(announce=False, **session.nd_params)

        # Avoid slewing if we are already on target
        if not session.on_target(target):
            user_logger.info('slewing to target')
            # Start moving each antenna to the target
            ants.req.mode('POINT')
            # Wait until they are all in position (with 5 minute timeout)
            ants.wait('lock', True, 300)
            user_logger.info('target reached')

            session.fire_noise_diode(announce=False, **session.nd_params)

        user_logger.info('tracking target')
        # Do nothing else for the duration of the track
        time.sleep(duration)
        user_logger.info('target tracked for %g seconds' % (duration,))

        session.fire_noise_diode(announce=False, **session.nd_params)
        return True

    @dynamic_doc("', '".join(_projections), _default_proj)
    def scan(self, target, duration=30.0, start=-3.0, end=3.0, offset=0.0, index=-1, scan_in_azimuth=True,
             projection=_default_proj, drive_strategy='shortest-slew', label='scan', announce=True):
        """Scan across a target.

        This scans across a target with all antennas involved in the session,
        either in azimuth or elevation (depending on the *scan_in_azimuth* flag).
        The scan starts at an offset of *start* degrees from the target and ends
        at an offset of *end* degrees along the scanning coordinate, while
        remaining at an offset of *offset* degrees from the target along the
        non-scanning coordinate. These offsets are calculated in a projected
        coordinate system (see *Notes* below). The scan lasts for *duration*
        seconds.

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target to scan across, as an object or description string
        duration : float, optional
            Minimum duration of scan across target, in seconds
        start : float, optional
            Start offset of scan position along scanning coordinate, in degrees
            (see *Notes* below)
        end : float, optional
            End offset of scan position along scanning coordinate, in degrees
            (see *Notes* below)
        offset : float, optional
            Offset of scan position along non-scanning coordinate, in degrees
        index : integer, optional
            Scan index, used for display purposes when this is part of a raster
        scan_in_azimuth : {True, False}, optional
            True if azimuth changes during scan while elevation remains fixed;
            False if scanning in elevation and stepping in azimuth instead
        projection : {'%s'}, optional
            Name of projection in which to perform scan relative to target
            (default = '%s')
        drive_strategy : {'shortest-slew', 'longest-track'}, optional
            Drive strategy employed by antennas, used to decide what to do when
            target is in azimuth overlap region of antenna. The default is to
            go to the wrap that is nearest to the antenna's current position,
            thereby saving time.
        label : string, optional
            Label associated with compound scan in HDF5 file, usually single word.
        announce : {True, False}, optional
            True if start of action should be announced, with details of settings

        Returns
        -------
        success : {True, False}
            True if scan was successfully completed

        Notes
        -----
        Take note that scanning is done in a projection on the celestial sphere,
        and the scan start and end are in the projected coordinates. The azimuth
        coordinate of a scan in azimuth will therefore change more than the
        *start* and *end* parameters suggest, especially at high elevations.
        This ensures that the same scan parameters will lead to the same
        qualitative scan for any position on the celestial sphere.

        When the function returns, the antennas will still track the end-point of
        the scan and data will still be recorded to the HDF5 file. The specified
        *duration* is therefore a minimum value. Remember to run :meth:`end` to
        close the file and finally stop the observation (automatically done when
        this object is used in a with-statement)!

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        scan_name = 'scan' if index < 0 else 'scan %d' % (index,)

        if announce:
            user_logger.info("Initiating %g-second scan across target '%s'" % (duration, target.name))
        if not session.target_visible(target, duration):
            user_logger.warning("Skipping scan, as target '%s' will be below horizon" % (target.name,))
            return False

        # Set the drive strategy for how antenna moves between targets, and the target
        ants.req.drive_strategy(drive_strategy)
        session.set_target(target)
        session.label(label)

        session.fire_noise_diode(announce=False, **session.nd_params)

        user_logger.info('slewing to start of %s' % (scan_name,))
        # Move each antenna to the start position of the scan
        if scan_in_azimuth:
            ants.req.scan_asym(start, offset, end, offset, duration, projection)
        else:
            ants.req.scan_asym(offset, start, offset, end, duration, projection)
        ants.req.mode('POINT')
        # Wait until they are all in position (with 5 minute timeout)
        ants.wait('lock', True, 300)
        user_logger.info('start of %s reached' % (scan_name,))

        session.fire_noise_diode(announce=False, **session.nd_params)

        user_logger.info('starting scan')
        # Start scanning the antennas
        ants.req.mode('SCAN')
        # Wait until they are all finished scanning (with 5 minute timeout)
        ants.wait('scan_status', 'after', 300)
        user_logger.info('%s complete' % (scan_name,))

        session.fire_noise_diode(announce=False, **session.nd_params)
        return True

    @dynamic_doc("', '".join(_projections), _default_proj)
    def raster_scan(self, target, num_scans=3, scan_duration=30.0, scan_extent=6.0, scan_spacing=0.5,
                    scan_in_azimuth=True, projection=_default_proj, drive_strategy='shortest-slew',
                    label='raster', announce=True):
        """Perform raster scan on target.

        A *raster scan* is a series of scans across a target performed by all
        antennas involved in the session, scanning in either azimuth or
        elevation while the other coordinate is changed in steps for each scan.
        Each scan is offset by the same amount on both sides of the target along
        the scanning coordinate (and therefore has the same extent), and the
        scans are arranged symmetrically around the target in the non-scanning
        (stepping) direction. If an odd number of scans are done, the middle
        scan will therefore pass directly over the target. The default is to
        scan in azimuth and step in elevation, leading to a series of horizontal
        scans. Each scan is scanned in the opposite direction to the previous
        scan to save time. Additionally, the first scan always starts at the top
        left of the target, regardless of scan direction.

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target to scan across, as an object or description string
        num_scans : integer, optional
            Number of scans across target (an odd number is better, as this will
            scan directly over the source during the middle scan)
        scan_duration : float, optional
            Minimum duration of each scan across target, in seconds
        scan_extent : float, optional
            Extent (angular length) of scan along scanning coordinate, in degrees
            (see *Notes* below)
        scan_spacing : float, optional
            Separation between each consecutive scan along the coordinate that is
            not scanned but stepped, in degrees
        scan_in_azimuth : {True, False}
            True if azimuth changes during scan while elevation remains fixed;
            False if scanning in elevation and stepping in azimuth instead
        projection : {'%s'}, optional
            Name of projection in which to perform scan relative to target
            (default = '%s')
        drive_strategy : {'shortest-slew', 'longest-track'}
            Drive strategy employed by antennas, used to decide what to do when
            target is in azimuth overlap region of antenna. The default is to
            go to the wrap that is nearest to the antenna's current position,
            thereby saving time.
        label : string, optional
            Label associated with compound scan in HDF5 file, usually single word.
        announce : {True, False}, optional
            True if start of action should be announced, with details of settings

        Returns
        -------
        success : {True, False}
            True if raster scan was successfully completed

        Notes
        -----
        Take note that scanning is done in a projection on the celestial sphere,
        and the scan extent and spacing apply to the projected coordinates.
        The azimuth coordinate of a scan in azimuth will therefore change more
        than the *scan_extent* parameter suggests, especially at high elevations.
        This ensures that the same scan parameters will lead to the same
        qualitative scan for any position on the celestial sphere.

        When the function returns, the antennas will still track the end-point of
        the last scan and data will still be recorded to the HDF5 file. The
        specified *scan_duration* is therefore a minimum value. Remember to run
        :meth:`end` to close the files and finally stop the observation
        (automatically done when this object is used in a with-statement)!

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)

        if announce:
            user_logger.info("Initiating raster scan (%d %g-second scans extending %g degrees) on target '%s'" %
                             (num_scans, scan_duration, scan_extent, target.name))
        # Calculate average time that noise diode is operated per scan, to add to scan duration in check below
        nd_time = session.nd_params['on'] + session.nd_params['off']
        nd_time *= scan_duration / max(session.nd_params['period'], scan_duration)
        nd_time = nd_time if session.nd_params['period'] >= 0 else 0.
        # Check whether the target will be visible for entire duration of raster scan
        if not session.target_visible(target, (scan_duration + nd_time) * num_scans):
            user_logger.warning("Skipping raster scan, as target '%s' will be below horizon" % (target.name,))
            return False

        # Create start positions of each scan, based on scan parameters
        scan_levels = np.arange(-(num_scans // 2), num_scans // 2 + 1)
        scanning_coord = (scan_extent / 2.0) * (-1) ** scan_levels
        stepping_coord = scan_spacing * scan_levels
        # Flip sign of elevation offsets to ensure that the first scan always starts at the top left of target
        scan_step = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(-scanning_coord, stepping_coord)

        # Perform multiple scans across the target
        for scan_index, (scan, step) in enumerate(scan_step):
            session.scan(target, duration=scan_duration, start=scan, end=-scan, offset=step, index=scan_index,
                         scan_in_azimuth=scan_in_azimuth, projection=projection, drive_strategy=drive_strategy,
                         label=label, announce=False)
        return True

    def end(self):
        """End the session, which stops data capturing and closes the data file.

        This does not affect the antennas, which continue to perform their
        last action (unless explicitly asked to stow).

        """
        # Create reference to session and KAT objects, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants

        # Obtain the name of the file currently being written to
        reply = kat.dbe.req.k7w_get_current_file()
        outfile = reply[1].replace('writing', 'unaugmented') if reply.succeeded else '<unknown file>'
        user_logger.info('Scans complete, data captured to %s' % (outfile,))

        # Stop the DBE data flow (this indirectly stops k7writer via a stop packet, but the HDF5 file is left open)
        kat.dbe.req.dbe_capture_stop()
        user_logger.info('Ended data capturing session with experiment ID %s' % (session.experiment_id,))
        kat.dbe.req.k7w_set_script_param('script-endtime', time.time())

        if session.stow_when_done:
            user_logger.info('stowing dishes')
            ants.req.mode('STOW')

        user_logger.info('==========================')

        # Disable logging to HDF5 file
        user_logger.removeHandler(self._script_log_handler)
        # Finally close the HDF5 file and prepare for augmentation after all logging and parameter settings are done
        kat.dbe.req.k7w_capture_done()

class TimeSession(object):
    """Fake CaptureSession object used to estimate the duration of an experiment."""
    def __init__(self, kat, experiment_id, observer, description, ants, stow_when_done=False, **kwargs):
        self.kat = kat
        self.experiment_id = experiment_id
        self.ants = []
        for ant in ant_array(kat, ants).devs:
            try:
                self.ants.append((katpoint.Antenna(ant.sensor.observer.get_value()),
                                  ant.sensor.mode.get_value(),
                                  ant.sensor.pos_actual_scan_azim.get_value(),
                                  ant.sensor.pos_actual_scan_elev.get_value()))
            except AttributeError:
                pass
        self.stow_when_done = stow_when_done
        # By default, no noise diodes are fired
        self.nd_params = {'diode' : 'coupler', 'on' : 0., 'off' : 0., 'period' : -1.}
        self.last_nd_firing = 0.

        self.start_time = time.time()
        self.time = self.start_time
        self.projection = ('ARC', 0., 0.)

        # Usurp time module functions that deal with the passage of real time, and connect them to session time instead
        self.realtime, self.realsleep = time.time, time.sleep
        time.time = lambda: self.time
        def simsleep(seconds):
            self.time += seconds
        time.sleep = simsleep

        # Modify logging so that only stream handlers are active and timestamps are prepended with a tilde
        for handler in user_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                form = handler.formatter
                form.old_datefmt = form.datefmt
                form.datefmt = '~ ' + (form.datefmt if form.datefmt else '%Y-%m-%d %H:%M:%S %Z')
            else:
                handler.old_level = handler.level
                handler.setLevel(100)

        user_logger.info('Estimating duration of experiment starting now (nothing real will happen!)')
        user_logger.info('==========================')
        user_logger.info('New data capturing session')
        user_logger.info('--------------------------')
        user_logger.info('Experiment ID = %s' % (experiment_id,))
        user_logger.info('Observer = %s' % (observer,))
        user_logger.info("Description ='%s'" % (description,))
        user_logger.info('Antennas used = %s' % (' '.join([ant[0].name for ant in self.ants]),))

    def __enter__(self):
        """Start time estimate, overriding the time module."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finish time estimate, restoring the time module."""
        self.end()
        # Do not suppress any exceptions that occurred in the body of with-statement
        return False

    def _azel(self, target, timestamp, antenna):
        """Target (az, el) position in degrees (including offsets in degrees)."""
        projection_type, x, y = self.projection
        az, el = target.plane_to_sphere(katpoint.deg2rad(x), katpoint.deg2rad(y), timestamp, antenna, projection_type)
        return katpoint.rad2deg(az), katpoint.rad2deg(el)

    def _teleport_to(self, target, mode='POINT'):
        """Move antennas instantaneously onto target (or nearest point on horizon)."""
        for m in range(len(self.ants)):
            antenna = self.ants[m][0]
            az, el = self._azel(target, self.time, antenna)
            self.ants[m] = (antenna, mode, az, max(el, 2.))

    def _slew_to(self, target, mode='POINT', timeout=300.):
        """Slew antennas to target (or nearest point on horizon), with timeout."""
        slew_times = []
        for ant, ant_mode, ant_az, ant_el in self.ants:
            def estimate_slew(timestamp):
                """Obtain instantaneous target position and estimate time to slew there."""
                # Target position right now
                az, el = self._azel(target, timestamp, ant)
                # If target is below horizon, aim at closest point on horizon
                az_dist, el_dist = np.abs(az - ant_az), np.abs(max(el, 2.) - ant_el)
                # Ignore azimuth wraps and drive strategies
                az_dist = az_dist if az_dist < 180. else 360. - az_dist
                # Assume az speed of 2 deg/s, el speed of 1 deg/s and overhead of 1 second
                slew_time = max(0.5 * az_dist, 1.0 * el_dist) + 1.0
                return az, el, slew_time
            # Initial estimate of slew time, based on a stationary target
            az1, el1, slew_time = estimate_slew(self.time)
            # Crude adjustment for target motion: chase target position for 2 iterations
            az2, el2, slew_time = estimate_slew(self.time + slew_time)
            az2, el2, slew_time = estimate_slew(self.time + slew_time)
            # Ensure slew does not take longer than timeout
            slew_time = min(slew_time, timeout)
            # If source is below horizon, handle timeout and potential rise in that interval
            if el2 < 2.:
                # Position after timeout
                az_after_timeout, el_after_timeout = self._azel(target, self.time + timeout, ant)
                # If source is still down, slew time == timeout, else estimate rise time through linear interpolation
                slew_time = (2. - el1) / (el_after_timeout - el1) * timeout if el_after_timeout > 2. else timeout
                az2, el2 = self._azel(target, self.time + slew_time, ant)
                el2 = max(el2, 2.)
            slew_times.append(slew_time)
#            print "%s slewing from (%.1f, %.1f) to (%.1f, %.1f) in %.1f seconds" % \
#                  (ant.name, ant_az, ant_el, az2, el2, slew_time)
        # The overall slew time is the max for all antennas - adjust current time to reflect the slew
        self.time += (np.max(slew_times) if len(slew_times) > 0 else 0.)
        # Blindly assume all antennas are on target (or on horizon) after this interval
        self._teleport_to(target, mode)

    def standard_setup(self, centre_freq=1800.0, dump_rate=1.0,
                       nd_params={'diode' : 'coupler', 'on' : 10.0, 'off' : 10.0, 'period' : 180.}, **kwargs):
        """Set up LO frequency, dump rate and noise diode parameters."""
        self.nd_params = nd_params
        user_logger.info('RF centre frequency = %g MHz, dump rate = %g Hz' % (centre_freq, dump_rate))
        if nd_params['period'] > 0:
            user_logger.info("Will switch '%s' noise diode on for %g s and off for %g s, every %g s if possible" %
                             (nd_params['diode'], nd_params['on'], nd_params['off'], nd_params['period']))
        elif nd_params['period'] == 0:
            user_logger.info("Will switch '%s' noise diode on for %g s and off for %g s at every opportunity" %
                             (nd_params['diode'], nd_params['on'], nd_params['off']))
        else:
            user_logger.info('Noise diode will not fire')
        user_logger.info('--------------------------')

    def on_target(self, target):
        """Determine whether antennas are tracking a given target."""
        for antenna, mode, ant_az, ant_el in self.ants:
            az, el = self._azel(target, self.time, antenna)
            # Checking for lock and checking for target identity considered the same thing
            if (az != ant_az) or (el != ant_el) or (mode != 'POINT'):
                return False
        return True

    def target_visible(self, target, duration=0., timeout=300., horizon=2., operation='scan'):
        """Check whether target is visible for given duration."""
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        horizon = katpoint.deg2rad(horizon)
        # Include an average time to slew to the target (worst case about 90 seconds, so half that)
        now = self.time + 45.
        average_el, visible_before, visible_after = [], [], []
        for antenna, mode, ant_az, ant_el in self.ants:
            az, el = target.azel(now, antenna)
            average_el.append(katpoint.rad2deg(el))
            # If not up yet, see if the target will pop out before the timeout
            if el < horizon:
                now += timeout
                az, el = target.azel(now, antenna)
            visible_before.append(el >= horizon)
            # Check what happens at end of observation
            az, el = target.azel(now + duration, antenna)
            visible_after.append(el >= horizon)
        if all(visible_before) and all(visible_after):
            return True
        always_invisible = any(~np.array(visible_before) & ~np.array(visible_after))
        if always_invisible:
            user_logger.warning("Target '%s' is never up during requested period (average elevation is %g degrees)" %
                                (target.name, np.mean(average_el)))
        else:
            user_logger.warning("Target '%s' will rise or set during requested period" % (target.name,))
        return False

    def start_scan(self, label):
        """Starting scan has no major timing effect."""
        pass

    def fire_noise_diode(self, diode='coupler', on=10.0, off=10.0, period=0.0, label='cal', announce=True):
        """Estimate time taken to fire noise diode."""
        if period < 0.0 or (self.time - self.last_nd_firing) < period:
            return False
        if announce:
            user_logger.info("Firing '%s' noise diode (%g seconds on, %g seconds off)" % (diode, on, off))
        else:
            user_logger.info('firing noise diode')
        self.time += on
        self.last_nd_firing = self.time + 0.
        self.time += off
        user_logger.info('fired noise diode')
        return True

    def track(self, target, duration=20.0, drive_strategy='longest-track', label='track', announce=True):
        """Estimate time taken to perform track."""
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        if announce:
            user_logger.info("Initiating %g-second track on target '%s'" % (duration, target.name))
        if not self.target_visible(target, duration):
            user_logger.warning("Skipping track, as target '%s' will be below horizon" % (target.name,))
            return False
        self.fire_noise_diode(label='', announce=False, **self.nd_params)
        if not self.on_target(target):
            user_logger.info('slewing to target')
            self._slew_to(target)
            user_logger.info('target reached')
            self.fire_noise_diode(announce=False, **self.nd_params)
        user_logger.info('tracking target')
        self.time += duration + 1.0
        user_logger.info('target tracked for %g seconds' % (duration,))
        self.fire_noise_diode(announce=False, **self.nd_params)
        self._teleport_to(target)
        return True

    def scan(self, target, duration=30.0, start=-3.0, end=3.0, scan_in_azimuth=True,
             projection=_default_proj, drive_strategy='shortest-slew', label='scan', announce=True):
        """Estimate time taken to perform single linear scan."""
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        if announce:
            user_logger.info("Initiating %g-second scan across target '%s'" % (duration, target.name))
        if not self.target_visible(target, duration):
            user_logger.warning("Skipping track, as target '%s' will be below horizon" % (target.name,))
            return False
        self.fire_noise_diode(label='', announce=False, **self.nd_params)
        projection = Offset.PROJECTIONS[projection]
        self.projection = (projection, start, 0.) if scan_in_azimuth else (projection, 0., start)
        user_logger.info('slewing to start of scan')
        self._slew_to(target, mode='SCAN')
        user_logger.info('start of scan reached')
        self.fire_noise_diode(announce=False, **self.nd_params)
        # Assume antennas can keep up with target (and doesn't scan too fast either)
        user_logger.info('starting scan')
        self.time += duration + 1.0
        user_logger.info('scan complete')
        self.fire_noise_diode(announce=False, **self.nd_params)
        self.projection = (projection, end, 0.) if scan_in_azimuth else (projection, 0., end)
        self._teleport_to(target)
        return True

    def raster_scan(self, target, num_scans=3, scan_duration=30.0, scan_extent=6.0, scan_spacing=0.5,
                    scan_in_azimuth=True, projection=_default_proj, drive_strategy='shortest-slew',
                    label='raster', announce=True):
        """Estimate time taken to perform raster scan."""
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        projection = Offset.PROJECTIONS[projection]
        if announce:
            user_logger.info("Initiating raster scan (%d %g-second scans extending %g degrees) on target '%s'" %
                             (num_scans, scan_duration, scan_extent, target.name))
        nd_time = self.nd_params['on'] + self.nd_params['off']
        nd_time *= scan_duration / max(self.nd_params['period'], scan_duration)
        nd_time = nd_time if self.nd_params['period'] >= 0 else 0.
        if not self.target_visible(target, (scan_duration + nd_time) * num_scans):
            user_logger.warning("Skipping track, as target '%s' will be below horizon" % (target.name,))
            return False
        # Create start positions of each scan, based on scan parameters
        scan_steps = np.arange(-(num_scans // 2), num_scans // 2 + 1)
        scanning_coord = (scan_extent / 2.0) * (-1) ** scan_steps
        stepping_coord = scan_spacing * scan_steps
        # These minus signs ensure that the first scan always starts at the top left of target
        scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)
        self.fire_noise_diode(label='', announce=False, **self.nd_params)
        # Iterate through the scans across the target
        for scan_count, scan in enumerate(scan_starts):
            self.projection = (projection, scan[0], scan[1])
            user_logger.info('slewing to start of scan %d' % (scan_count,))
            self._slew_to(target, mode='SCAN')
            user_logger.info('start of scan %d reached' % (scan_count,))
            self.fire_noise_diode(announce=False, **self.nd_params)
            # Assume antennas can keep up with target (and doesn't scan too fast either)
            user_logger.info('starting scan %d' % (scan_count,))
            self.time += scan_duration + 1.0
            user_logger.info('scan %d complete' % (scan_count,))
            self.fire_noise_diode(announce=False, **self.nd_params)
            self.projection = (projection, -scan[0], scan[1]) if scan_in_azimuth else (projection, scan[0], -scan[1])
            self._teleport_to(target)
        return True

    def end(self):
        """Stop data capturing to shut down the session and close the data file."""
        user_logger.info('Scans complete, no data captured as this is a timing simulation...')
        user_logger.info('Ended data capturing session with experiment ID %s' % (self.experiment_id,))
        if self.stow_when_done:
            user_logger.info("Stowing dishes.")
            self._teleport_to(katpoint.Target("azel, 0.0, 90.0"), mode="STOW")
        user_logger.info('==========================')
        duration = self.time - self.start_time
        if duration <= 100:
            duration = '%d seconds' % (np.ceil(duration),)
        elif duration <= 100 * 60:
            duration = '%d minutes' % (np.ceil(duration / 60.),)
        else:
            duration = '%.1f hours' % (duration / 3600.,)
        user_logger.info("Experiment estimated to last %s until this time\n" % (duration,))
        # Restore time module functions
        time.time, time.sleep = self.realtime, self.realsleep
        # Restore logging
        for handler in user_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.formatter.datefmt = handler.formatter.old_datefmt
                del handler.formatter.old_datefmt
            else:
                handler.setLevel(handler.old_level)
                del handler.old_level


def standard_script_options(usage, description):
    """Create option parser pre-populated with standard observation script options.

    Parameters
    ----------
    usage, description : string
        Usage and description strings to be used for script help

    Returns
    -------
    parser : :class:`optparse.OptionParser` object
        Parser populated with standard script options

    """
    parser = optparse.OptionParser(usage=usage, description=description)

    parser.add_option('-s', '--system', help='System configuration file to use, relative to conf directory '
                      '(default reuses existing connection, or falls back to systems/local.conf)')
    parser.add_option('-u', '--experiment-id', help='Experiment ID used to link various parts of experiment '
                      'together (UUID generated by default)')
    parser.add_option('-o', '--observer', help='Name of person doing the observation (**required**)')
    parser.add_option('-d', '--description', default='No description.',
                      help="Description of observation (default='%default')")
    parser.add_option('-a', '--ants', help="Comma-separated list of antennas to include " +
                      "(e.g. 'ant1,ant2'), or 'all' for all antennas (**required** - safety reasons)")
    parser.add_option('-f', '--centre-freq', type='float', default=1822.0,
                      help='Centre frequency, in MHz (default=%default)')
    parser.add_option('-r', '--dump-rate', type='float', default=1.0, help='Dump rate, in Hz (default=%default)')
    parser.add_option('-n', '--nd-params', default='coupler,10,10,180',
                      help="Noise diode parameters as 'diode,on,off,period', in seconds (default='%default')")
    parser.add_option('-p', '--projection', type='choice', choices=_projections, default=_default_proj,
                      help="Spherical projection in which to perform scans, one of '%s' (default), '%s'" %
                           (_projections[0], "', '".join(_projections[1:])))
    parser.add_option('-y', '--dry-run', action='store_true', default=False,
                      help="Do not actually observe, but display script actions at predicted times (default=%default)")
    parser.add_option('--stow-when-done', action='store_true', default=False,
                      help="Stow the antennas when the capture session ends.")

    return parser

def verify_and_connect(opts):
    """Verify command-line options, build KAT configuration and connect to devices.

    This inspects the parsed options and requires at least *ants* and *observer*
    to be set. It generates an experiment ID if missing and verifies noise diode
    parameters if given. It then creates a KAT connection based on the *system*
    option, reusing an existing connection or falling back to the local system
    if required. The resulting KATHost object is returned.

    Parameters
    ----------
    opts : :class:`optparse.Values` object
        Parsed command-line options (will be updated by this function). Should
        contain at least the options *ants*, *observer* and *system*.

    Returns
    -------
    kat : :class:`utility.KATHost` object
        KAT connection object associated with this experiment

    Raises
    ------
    ValueError
        If required options are missing

    """
    # Various non-optional options...
    if not hasattr(opts, 'ants') or opts.ants is None:
        raise ValueError('Please specify the antennas to use via -a option (yes, this is a non-optional option...)')
    if not hasattr(opts, 'observer') or opts.observer is None:
        raise ValueError('Please specify the observer name via -o option (yes, this is a non-optional option...)')
    if not hasattr(opts, 'experiment_id') or opts.experiment_id is None:
        # Generate unique string via RFC 4122 version 1
        opts.experiment_id = str(uuid.uuid1())

    # If given, verify noise diode parameters (should be 'string,number,number,number') and convert to dict
    if hasattr(opts, 'nd_params'):
        try:
            opts.nd_params = eval("{'diode':'%s', 'on':%s, 'off':%s, 'period':%s}" %
                                  tuple(opts.nd_params.split(',')), {})
        except (TypeError, NameError):
            raise ValueError("Noise diode parameters are incorrect (should be 'diode,on,off,period')")
        for key in ('on', 'off', 'period'):
            if opts.nd_params[key] != float(opts.nd_params[key]):
                raise ValueError("Parameter nd_params['%s'] = %s (should be a number)" % (key, opts.nd_params[key]))

    # Try to build KAT configuration (which might be None, in which case try to reuse latest active connection)
    # This connects to all the proxies and devices and queries their commands and sensors
    try:
        kat = tbuild(opts.system)
    # Fall back to *local* configuration to prevent inadvertent use of the real hardware
    except ValueError:
        kat = tbuild('systems/local.conf')
    user_logger.info("Using KAT connection with configuration: %s" % (kat.config_file,))

    return kat

def lookup_targets(kat, args):
    """Look up targets by name in default catalogue, or keep as description string.

    Parameters
    ----------
    kat : :class:`utility.KATHost` object
        KAT connection object associated with this experiment
    args : list of strings
        Argument list containing mixture of target names and description strings

    Returns
    -------
    targets : list of strings and :class:`katpoint.Target` objects
        Targets as objects or description strings

    Raises
    ------
    ValueError
        If final target list is empty

    """
    # Look up target names in catalogue, and keep target description strings as is
    targets = []
    for arg in args:
        # With no comma in the target string, assume it's the name of a target to be looked up in the standard catalogue
        if arg.find(',') < 0:
            target = kat.sources[arg]
            if target is None:
                user_logger.info("Unknown source '%s', skipping it" % (arg,))
            else:
                targets.append(target)
        else:
            # Assume the argument is a target description string
            targets.append(arg)
    if len(targets) == 0:
        raise ValueError("No known targets found")
    return targets
