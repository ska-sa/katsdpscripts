"""CaptureSession encompassing data capturing and standard observations with KAT.

This defines the :class:`CaptureSession` class, which encompasses the capturing
of data and the performance of standard scans with the Fringe Finder system. It
also provides a fake :class:`TimeSession` class, which goes through the motions
in order to time them, but without performing any real actions.

"""

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
import logging
import sys
import os.path

import numpy as np
import katpoint
# This is used to document available spherical projections (and set them in case of TimeSession)
from katcore.proxy.antenna_proxy import AntennaProxyModel, Offset

from .array import Array
from .katcp_client import KATDevice
from .defaults import user_logger, activity_logger
from .misc import dynamic_doc

# Obtain list of spherical projections and the default projection from antenna proxy
projections, default_proj = AntennaProxyModel.PROJECTIONS, AntennaProxyModel.DEFAULT_PROJECTION
# Move default projection to front of list
projections.remove(default_proj)
projections.insert(0, default_proj)

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

class CaptureScan(object):
    """Context manager that encapsulates the capturing of a single scan.

    Depending on the scan settings, this ensures that data capturing has started
    at the start of the scan and that a new Scan group has been created in the
    output HDF5 file. After the scan is done, the context manager ensures that
    capturing is paused again, if this is desired.

    Parameters
    ----------
    kat : :class:`utility.KATHost` object
        KAT connection object associated with this experiment
    label : string
        Label for scan in HDF5 file, usually a single computer-parseable word.
        If this is an empty string, do not create a new Scan group in the file.
    dbe : :class:`KATDevice` object
        DBE proxy device for the session
    record_slews : {True, False}
        If True, correlator data is recorded contiguously and the data file
        includes 'slew' scans which occur while the antennas slew to the start
        of the next proper scan. If False, the file output (but not the signal
        displays) is paused while the antennas slew, and the data file contains
        only proper scans.

    """
    def __init__(self, kat, label, dbe, record_slews):
        self.kat = kat
        self.label = label
        self.dbe = dbe
        self.record_slews = record_slews

    def __enter__(self):
        """Start with scan and start/unpause capturing if necessary."""
        # Do nothing if we want to slew and slews are not to be recorded
        if self.label == 'slew' and not self.record_slews:
            return self
        # Create new Scan group in HDF5 file, if label is non-empty
        if self.label:
            self.dbe.req.k7w_new_scan(self.label)
        # Unpause HDF5 file output (redundant if output is never paused anyway)
        self.dbe.req.k7w_write_hdf5(1)
        # If we haven't yet, start recording data from the correlator (which creates the data file)
        if self.dbe.sensor.capturing.get_value() == '0':
            self.dbe.req.capture_start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finish with scan and pause capturing if necessary."""
        # If slews are not to be recorded, pause the file output again afterwards
        if not self.record_slews:
            self.dbe.req.k7w_write_hdf5(0)
        # Do not suppress any exceptions that occurred in the body of with-statement
        return False

class CaptureSession(object):
    """Context manager that encapsulates a single data capturing session.

    A data capturing *session* results in a single data file, potentially
    containing multiple scans and compound scans. An *experiment* may consist of
    multiple sessions. This object ensures that the capturing process is
    started and completed cleanly, even if exceptions occur during the session.
    It also provides canned routines for simple observations such as tracks,
    single scans and raster scans on a specific source.

    The initialisation of the session object does basic preparation of the data
    capturing subsystem (k7writer) and logging. It tries to do the minimum to
    enable data capturing. The experimental setup is usually completed by
    calling :meth:`standard_setup` on the instantiated session object.
    The actual data capturing only starts once a canned routine is called.

    Parameters
    ----------
    kat : :class:`utility.KATHost` object
        KAT connection object associated with this experiment
    dbe : :class:`KATDevice` object or string, optional
        DBE proxy to use (effectively selects the correlator)
    kwargs : dict, optional
        Ignore any other keyword arguments (simplifies passing options as dict)

    """
    def __init__(self, kat, dbe='dbe', **kwargs):
        try:
            self.kat = kat
            # If not a device itself, assume dbe is the name of the device
            if not isinstance(dbe, KATDevice):
                try:
                    dbe = getattr(kat, dbe)
                except AttributeError:
                    raise ValueError("DBE proxy '%s' not found (i.e. no kat.%s exists)" % (dbe, dbe))
            self.dbe = dbe

            # Default settings for session parameters (in case standard_setup is not called)
            self.ants = None
            self.experiment_id = 'interactive'
            self.record_slews = True
            self.stow_when_done = False
            self.nd_params = {'diode' : 'coupler', 'on' : 0., 'off' : 0., 'period' : -1.}
            self.last_nd_firing = 0.
            self.output_file = ''

            activity_logger.info("----- Script starting  %s (%s)" % (sys.argv[0], ' '.join(sys.argv[1:])))

            user_logger.info("==========================")
            user_logger.info("New data capturing session")
            user_logger.info("--------------------------")
            user_logger.info("DBE proxy used = %s" % (dbe.name,))

            # Start with a clean state, by stopping the DBE
            dbe.req.capture_stop()
            # Set data output directory (typically on ff-dc machine)
            dbe.req.k7w_output_directory("/var/kat/data")
            # Enable output to HDF5 file (takes effect on capture_start only)
            dbe.req.k7w_write_hdf5(1)

            # Log the script parameters (if config manager is around)
            if kat.has_connected_device('cfg'):
                kat.cfg.req.set_script_param("script-starttime",
                                             time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time())))
                kat.cfg.req.set_script_param("script-endtime", "")
                kat.cfg.req.set_script_param("script-name", sys.argv[0])
                kat.cfg.req.set_script_param("script-arguments", ' '.join(sys.argv[1:]))
                kat.cfg.req.set_script_param("script-status", "busy")
        except Exception, e:
            if kat.has_connected_device('cfg'):
                kat.cfg.req.set_script_param("script-endtime",
                                             time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time())))
                kat.cfg.req.set_script_param("script-status", "failed")
            msg = "CaptureSession failed to initialise (%s)" % (e,)
            user_logger.error(msg)
            activity_logger.error(msg)
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

    def standard_setup(self, ants, observer, description, experiment_id=None, centre_freq=None,
                       dump_rate=1.0, nd_params=None, record_slews=None, stow_when_done=None, **kwargs):
        """Perform basic experimental setup including antennas, LO and dump rate.

        This performs the basic high-level setup that most experiments require.
        It should usually be called as the first step in a new session
        (unless the experiment has special requirements, such as holography).

        The user selects a subarray of antennas that will take part in the
        experiment, identifies him/herself and describes the experiment.
        Optionally, the user may also set the RF centre frequency, dump rate
        and noise diode firing strategy, amongst others. All optional settings
        are left unchanged if unspecified, except for the dump rate, which has
        to be set (due to the fact that there is currently no way to determine
        the dump rate...).

        The antenna specification *ants* do not have a default, which forces the
        user to specify them explicitly. This is for safety reasons, to remind
        the user of which antennas will be moved around by the script. The
        *observer* and *description* similarly have no default, to force the
        user to document the observation to some extent.

        Parameters
        ----------
        ants : :class:`Array` or :class:`KATDevice` object, or list, or string
            Antennas that will participate in the capturing session, as an Array
            object containing antenna devices, or a single antenna device or a
            list of antenna devices, or a string of comma-separated antenna
            names, or the string 'all' for all antennas controlled via the
            KAT connection associated with this session
        observer : string
            Name of person doing the observation
        description : string
            Short description of the purpose of the capturing session
        experiment_id : string, optional
            Experiment ID, a unique string used to link the data files of an
            experiment together with blog entries, etc. (unchanged by default)
        centre_freq : float, optional
            RF centre frequency, in MHz (unchanged by default)
        dump_rate : float, optional
            Correlator dump rate, in Hz (will be set by default)
        nd_params : dict, optional
            Dictionary containing parameters that control firing of the noise
            diode during canned commands. These parameters are in the form of
            keyword-value pairs, and matches the parameters of the
            :meth:`fire_noise_diode` method. This is unchanged by default
            (typically disabling automatic firing).
        record_slews : {True, False}, optional
            If True, correlator data is recorded contiguously and the data file
            includes 'slew' scans which occur while the antennas slew to the
            start of the next proper scan. If False, the file output (but not
            the signal displays) is paused while the antennas slew, and the data
            file contains only proper scans. This is unchanged by default.
        stow_when_done : {False, True}, optional
            If True, stow the antennas when the capture session completes
            (unchanged by default)
        kwargs : dict, optional
            Ignore any other keyword arguments (simplifies passing options as dict)

        Raises
        ------
        ValueError
            If antenna with a specified name is not found on KAT connection object

        """
        # Create references to allow easy copy-and-pasting from this function
        session, kat, dbe = self, self.kat, self.dbe

        session.ants = ants = ant_array(kat, ants)
        # Override provided session parameters (or initialize them from existing parameters if not provided)
        session.experiment_id = experiment_id = session.experiment_id if experiment_id is None else experiment_id
        session.nd_params = nd_params = session.nd_params if nd_params is None else nd_params
        session.record_slews = record_slews = session.record_slews if record_slews is None else record_slews
        session.stow_when_done = stow_when_done = session.stow_when_done if stow_when_done is None else stow_when_done

        # Setup strategies for the sensors we might be wait()ing on
        ants.req.sensor_sampling("lock", "event")
        ants.req.sensor_sampling("scan.status", "event")
        ants.req.sensor_sampling("mode", "event")

        # Setup basic experiment info
        dbe.req.k7w_experiment_info(experiment_id, observer, description)

        # Set centre frequency in RFE stage 7 (and read it right back to verify)
        if centre_freq is not None:
            kat.rfe7.req.rfe7_lo1_frequency(4200.0 + centre_freq, 'MHz')
        centre_freq = kat.rfe7.sensor.rfe7_lo1_frequency.get_value() * 1e-6 - 4200.0
        effective_lo_freq = (centre_freq - 200.0) * 1e6
        # The DBE proxy needs to know the dump period (in ms) as well as the effective LO freq,
        # which is used for fringe stopping (eventually). This sets the delay model and other
        # correlator parameters, such as the dump rate, and instructs the correlator to pass
        # its data to the k7writer daemon (set via configuration)
        dbe.req.capture_setup(1000.0 / dump_rate, effective_lo_freq)

        user_logger.info("Antennas used = %s" % (' '.join([ant.name for ant in ants.devs]),))
        user_logger.info("Observer = %s" % (observer,))
        user_logger.info("Description ='%s'" % (description,))
        user_logger.info("Experiment ID = %s" % (experiment_id,))
        user_logger.info("RF centre frequency = %g MHz, dump rate = %g Hz, keep slews = %s" %
                         (centre_freq, dump_rate, record_slews))
        if nd_params['period'] > 0:
            nd_info = "Will switch '%s' noise diode on for %g s and off for %g s, every %g s if possible" % \
                      (nd_params['diode'], nd_params['on'], nd_params['off'], nd_params['period'])
        elif nd_params['period'] == 0:
            nd_info = "Will switch '%s' noise diode on for %g s and off for %g s at every opportunity" % \
                      (nd_params['diode'], nd_params['on'], nd_params['off'])
        else:
            nd_info = "Noise diode will not fire automatically"
        user_logger.info(nd_info + " while performing canned commands")

        # Log the script parameters (if config manager is around)
        if kat.has_connected_device('cfg'):
            kat.cfg.req.set_script_param("script-observer", observer)
            kat.cfg.req.set_script_param("script-description", description)
            kat.cfg.req.set_script_param("script-experiment-id", experiment_id)
            kat.cfg.req.set_script_param("script-rf-params", "Freq=%g MHz, Dump rate=%g Hz, Keep slews=%s" %
                                                             (centre_freq, dump_rate, record_slews))
            kat.cfg.req.set_script_param("script-nd-params", "Diode=%s, On=%g s, Off=%g s, Period=%g s" %
                                         (nd_params['diode'], nd_params['on'], nd_params['off'], nd_params['period']))

        # If the DBE is simulated, it will have position update commands
        if hasattr(dbe.req, 'dbe_pointing_az') and hasattr(dbe.req, 'dbe_pointing_el'):
            first_ant = ants.devs[0]
            # The minimum time between position updates is fraction of dump period to ensure fresh data at every dump
            update_period_seconds = 0.4 / dump_rate
            # Tell the position sensors to report their values periodically at this rate
            # Remember that this should be an *integer* number of milliseconds
            first_ant.sensor.pos_actual_scan_azim.set_strategy('period', str(update_period_seconds))
            first_ant.sensor.pos_actual_scan_elev.set_strategy('period', str(update_period_seconds))
            # Tell the DBE simulator where the first antenna is so that it can generate target flux at the right time
            first_ant.sensor.pos_actual_scan_azim.register_listener(dbe.req.dbe_pointing_az, update_period_seconds)
            first_ant.sensor.pos_actual_scan_elev.register_listener(dbe.req.dbe_pointing_el, update_period_seconds)
            user_logger.info("DBE simulator receives position updates from antenna '%s'" % (first_ant.name,))
        user_logger.info("--------------------------")

    def capture_start(self):
        """Start capturing data (ignored in version 1, as start is implicit)."""
        pass

    def label(self, label):
        """Add timestamped label to HDF5 file (ignored in version 1)."""
        pass

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
        if self.ants is None:
            return False
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
        if self.ants is None:
            return False
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

    def start_scan(self, label):
        """Set up start and shutdown of scan (the basic unit of an experiment).

        This returns a *context manager* to be used in a *with* statement, which
        controls the creation of a new Scan group in the output HDF5 file and
        pauses and unpauses data recording, as required. It should be used to
        delimit all the actions involving a single scan.

        Parameters
        ----------
        label : string
            Label for scan in HDF5 file, usually single computer-parseable word.
            If this is an empty string, do not create new Scan group in the file.

        Returns
        -------
        capture_scan : :class:`CaptureScan` object
            Context manager that encapsulates capturing of a single scan

        """
        return CaptureScan(self.kat, label, self.dbe, self.record_slews)

    def fire_noise_diode(self, diode='pin', on=10.0, off=10.0, period=0.0, label='cal', announce=True):
        """Switch noise diode on and off.

        This switches the selected noise diode on and off for all the antennas
        doing the observation. If a label is provided, a new Scan group is
        created in the HDF5 file. The target and compound scan are not changed.

        The on and off durations can be specified. Additionally, setting the
        *period* allows the noise diode to be fired on a semi-regular basis. The
        diode will only be fired if more than *period* seconds have elapsed since
        the last firing. If *period* is 0, the diode is fired unconditionally.
        On the other hand, if *period* is negative it is not fired at all.

        When the function returns, data will still be recorded to the HDF5 file.
        The specified *off* duration is therefore a minimum value. Remember to
        run :meth:`shutdown` to close the file and finally stop the observation
        (automatically done when this object is used in a with-statement)!

        Parameters
        ----------
        diode : {'pin', 'coupler'}
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
        label : string, optional
            Label for scan in HDF5 file, usually single computer-parseable word.
            If this is an empty string, do not create new Scan group in the file.
        announce : {True, False}, optional
            True if start of action should be announced, with details of settings

        Returns
        -------
        fired : {True, False}
            True if noise diode fired

        """
        if self.ants is None:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
        # Create references to allow easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants

        # If period is non-negative, quit if it is not yet time to fire the noise diode
        if period < 0.0 or (time.time() - session.last_nd_firing) < period:
            return False

        if announce:
            user_logger.info("Firing '%s' noise diode (%g seconds on, %g seconds off)" % (diode, on, off))
        else:
            user_logger.info('firing noise diode')

        with session.start_scan(label):
            # Switch noise diode on on all antennas
            ants.req.rfe3_rfe15_noise_source_on(diode, 1, 'now', 0)
            time.sleep(on)
            # Mark on -> off transition as last firing
            session.last_nd_firing = time.time()
            # Switch noise diode off on all antennas
            ants.req.rfe3_rfe15_noise_source_on(diode, 0, 'now', 0)
            time.sleep(off)
        user_logger.info('noise diode fired')

        return True

    def track(self, target, duration=20.0, drive_strategy='shortest-slew', label='track', announce=True):
        """Track a target.

        This tracks the specified target while recording data.

        In addition to the proper track on the source (labelled 'scan' in the
        dataset), data may also be recorded while the antennas are moving to the
        start of the track. This segment is labelled 'slew' in the dataset and
        will typically be discarded during processing.

        Data capturing is started before the track starts, if it isn't running
        yet. If a label or a new target is supplied, a new compound scan will be
        created in the HDF5 data file, with an optional 'slew' scan followed by
        a 'scan' scan. The antennas all track the same target in parallel.

        When the function returns, the antennas will still track the target and
        data will still be recorded to the HDF5 file. The specified *duration*
        is therefore a minimum value. Remember to run :meth:`shutdown` to close
        the file and finally stop the observation (automatically done when this
        object is used in a with-statement)!

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target to track, as an object or description string
        duration : float, optional
            Minimum duration of track, in seconds
        drive_strategy : {'shortest-slew', 'longest-track'}, optional
            Drive strategy employed by antennas, used to decide what to do when
            target is in azimuth overlap region of antenna. The default is to
            go to the wrap that will permit the longest possible track before
            the target sets.
        label : string, optional
            Label for compound scan in HDF5 file, usually a single word. If this
            is an empty string and *target* matches the target of the current
            compound scan being written, do not create new CompoundScan group
            in the file.
        announce : {True, False}, optional
            True if start of action should be announced, with details of settings

        Returns
        -------
        success : {True, False}
            True if track was successfully completed

        """
        if self.ants is None:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
        # Create references to allow easy copy-and-pasting from this function
        session, kat, ants, dbe = self, self.kat, self.ants, self.dbe
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)

        if announce:
            user_logger.info("Initiating %g-second track on target '%s'" % (duration, target.name))
        if not session.target_visible(target, duration):
            user_logger.warning("Skipping track, as target '%s' will be below horizon" % (target.name,))
            return False
        # Check if we are currently on the desired target (saves a slewing step)
        on_target = session.on_target(target)

        # Set the drive strategy for how antenna moves between targets
        ants.req.drive_strategy(drive_strategy)
        # Set the antenna target (antennas will already move there if in mode 'POINT')
        ants.req.target(target)
        # Provide target to the DBE proxy, which will use it as delay-tracking center
        dbe.req.target(target)
        # If using DBE simulator and target is azel type, move test target here (allows changes in correlation power)
        if hasattr(dbe.req, 'dbe_test_target') and target.body_type == 'azel':
            azel = katpoint.rad2deg(np.array(target.azel()))
            dbe.req.dbe_test_target(azel[0], azel[1], 100.)
        # Obtain target associated with the current compound scan
        req = dbe.req.k7w_get_target()
        current_target = req[1] if req else ''
        # Ensure that there is a label if a new compound scan is forced
        if target != current_target and not label:
            label = 'track'

        # If desired, create new CompoundScan group in HDF5 file, which automatically also creates the first Scan group
        if label:
            dbe.req.k7w_new_compound_scan(target, label, 'cal')
            session.fire_noise_diode(label='', announce=False, **session.nd_params)
        else:
            session.fire_noise_diode(announce=False, **session.nd_params)

        # Avoid slewing if we are already on target
        if not on_target:
            user_logger.info('slewing to target')
            with session.start_scan('slew'):
                # Start moving each antenna to the target
                ants.req.mode('POINT')
                # Wait until they are all in position (with 5 minute timeout)
                ants.wait('lock', True, 300)
            user_logger.info('target reached')

            session.fire_noise_diode(announce=False, **session.nd_params)

        user_logger.info('tracking target')
        with session.start_scan('scan'):
            # Do nothing else for the duration of the track
            time.sleep(duration)
        user_logger.info('target tracked for %g seconds' % (duration,))

        session.fire_noise_diode(announce=False, **session.nd_params)
        return True

    @dynamic_doc("', '".join(projections), default_proj)
    def scan(self, target, duration=30.0, start=(-3.0, 0.0), end=(3.0, 0.0), index=-1,
             projection=default_proj, drive_strategy='shortest-slew', label='scan', announce=True):
        """Scan across a target.

        This scans across a target, either in azimuth or elevation (depending on
        the *scan_in_azimuth* flag). The scan starts at an offset of *start*
        degrees from the target and ends at an offset of *end* degrees. These
        offsets are calculated in a projected coordinate system (see *Notes*
        below). The scan lasts for *duration* seconds.

        In addition to the proper scan across the source (labelled 'scan' in the
        dataset), data may also be recorded while the antennas are moving to the
        start of the scan. This segment is labelled 'slew' in the dataset and
        will typically be discarded during processing.

        Data capturing is started before the scan starts, if it isn't running yet.
        If a label or a new target is supplied, a new compound scan will be
        created in the HDF5 data file, with an optional 'slew' scan followed by
        a 'scan' scan. The antennas all scan across the same target in parallel.

        When the function returns, the antennas will still track the end-point of
        the scan and data will still be recorded to the HDF5 file. The specified
        *duration* is therefore a minimum value. Remember to run :meth:`shutdown`
        to close the file and finally stop the observation (automatically done
        when this object is used in a with-statement)!

        Parameters
        ----------
        target : :class:`katpoint.Target` object or string
            Target to scan across, as an object or description string
        duration : float, optional
            Minimum duration of scan across target, in seconds
        start : sequence of 2 floats, optional
            Initial scan position as (x, y) offset in degrees (see *Notes* below)
        end : sequence of 2 floats, optional
            Final scan position as (x, y) offset in degrees (see *Notes* below)
        index : integer, optional
            Scan index, used for display purposes when this is part of a raster
        projection : {'%s'}, optional
            Name of projection in which to perform scan relative to target
            (default = '%s')
        drive_strategy : {'shortest-slew', 'longest-track'}, optional
            Drive strategy employed by antennas, used to decide what to do when
            target is in azimuth overlap region of antenna. The default is to
            go to the wrap that is nearest to the antenna's current position,
            thereby saving time.
        label : string, optional
            Label for compound scan in HDF5 file, usually a single word. If this
            is an empty string and *target* matches the target of the current
            compound scan being written, do not create new CompoundScan group
            in the file.
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
        *start* and *end* parameters suggest, especially at high elevations
        (unless the 'plate-carree' projection is used). This ensures that the
        same scan parameters will lead to the same qualitative scan for any
        position on the celestial sphere.

        """
        if self.ants is None:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
        # Create references to allow easy copy-and-pasting from this function
        session, kat, ants, dbe = self, self.kat, self.ants, self.dbe
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        scan_name = 'scan' if index < 0 else 'scan %d' % (index,)

        if announce:
            user_logger.info("Initiating %g-second scan across target '%s'" % (duration, target.name))
        if not session.target_visible(target, duration):
            user_logger.warning("Skipping scan, as target '%s' will be below horizon" % (target.name,))
            return False

        # Set the drive strategy for how antenna moves between targets
        ants.req.drive_strategy(drive_strategy)
        # Set the antenna target
        ants.req.target(target)
        # Provide target to the DBE proxy, which will use it as delay-tracking center
        dbe.req.target(target)
        # If using DBE simulator and target is azel type, move test target here (allows changes in correlation power)
        if hasattr(dbe.req, 'dbe_test_target') and target.body_type == 'azel':
            azel = katpoint.rad2deg(np.array(target.azel()))
            dbe.req.dbe_test_target(azel[0], azel[1], 100.)
        # Obtain target associated with the current compound scan
        req = dbe.req.k7w_get_target()
        current_target = req[1] if req else ''
        # Ensure that there is a label if a new compound scan is forced
        if target != current_target and not label:
            label = 'scan'

        # If desired, create new CompoundScan group in HDF5 file, which automatically also creates the first Scan group
        if label:
            dbe.req.k7w_new_compound_scan(target, label, 'cal')
            session.fire_noise_diode(label='', announce=False, **session.nd_params)
        else:
            session.fire_noise_diode(announce=False, **session.nd_params)

        user_logger.info('slewing to start of %s' % (scan_name,))
        with session.start_scan('slew'):
            # Move each antenna to the start position of the scan
            ants.req.scan_asym(start[0], start[1], end[0], end[1], duration, projection)
            ants.req.mode('POINT')
            # Wait until they are all in position (with 5 minute timeout)
            ants.wait('lock', True, 300)
        user_logger.info('start of %s reached' % (scan_name,))

        session.fire_noise_diode(announce=False, **session.nd_params)

        user_logger.info('performing %s' % (scan_name,))
        with session.start_scan('scan'):
            # Start scanning the antennas
            ants.req.mode('SCAN')
            # Wait until they are all finished scanning (with 5 minute timeout)
            ants.wait('scan_status', 'after', 300)
        user_logger.info('%s complete' % (scan_name,))

        session.fire_noise_diode(announce=False, **session.nd_params)
        return True

    @dynamic_doc("', '".join(projections), default_proj)
    def raster_scan(self, target, num_scans=3, scan_duration=30.0, scan_extent=6.0, scan_spacing=0.5,
                    scan_in_azimuth=True, projection=default_proj, drive_strategy='shortest-slew',
                    label='raster', announce=True):
        """Perform raster scan on target.

        A *raster scan* is a series of scans across a target, scanning in either
        azimuth or elevation, while the other coordinate is changed in steps for
        each scan. Each scan is offset by the same amount on both sides of the
        target along the scanning coordinate (and therefore has the same extent),
        and the scans are arranged symmetrically around the target in the
        non-scanning (stepping) direction. If an odd number of scans are done,
        the middle scan will therefore pass directly over the target. The default
        is to scan in azimuth and step in elevation, leading to a series of
        horizontal scans. Each scan is scanned in the opposite direction to the
        previous scan to save time. Additionally, the first scan always starts
        at the top left of the target, regardless of scan direction.

        In addition to the proper scans across the source (labelled 'scan' in the
        dataset), data may also be recorded while the antennas are moving to the
        start of the next scan. These segments are labelled 'slew' and will
        typically be discarded during processing.

        Data capturing is started before the first scan, if it isn't running yet.
        All scans in the raster scan are grouped together in a single compound
        scan in the HDF5 data file, as they share the same target. If a label or
        a new target is supplied, a new compound scan will be created, otherwise
        the existing one will be re-used.The antennas all perform the same raster
        scan across the given target, in parallel.

        When the function returns, the antennas will still track the end-point of
        the last scan and data will still be recorded to the HDF5 file. The
        specified *scan_duration* is therefore a minimum value. Remember to run
        :meth:`shutdown` to close the files and finally stop the observation
        (automatically done when this object is used in a with-statement)!

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
            Label for compound scan in HDF5 file, usually a single word. If this
            is an empty string and *target* matches the target of the current
            compound scan being written, do not create new CompoundScan group
            in the file.
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

        """
        if self.ants is None:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
        # Create references to allow easy copy-and-pasting from this function
        session, kat, ants, dbe = self, self.kat, self.ants, self.dbe
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

        # Set the drive strategy for how antenna moves between targets
        ants.req.drive_strategy(drive_strategy)
        # Set the antenna target
        ants.req.target(target)
        # Provide target to the DBE proxy, which will use it as delay-tracking center
        dbe.req.target(target)
        # If using DBE simulator and target is azel type, move test target here (allows changes in correlation power)
        if hasattr(dbe.req, 'dbe_test_target') and target.body_type == 'azel':
            azel = katpoint.rad2deg(np.array(target.azel()))
            dbe.req.dbe_test_target(azel[0], azel[1], 100.)
        # Obtain target associated with the current compound scan
        req = dbe.req.k7w_get_target()
        current_target = req[1] if req else ''
        # Ensure that there is a label if a new compound scan is forced
        if target.description != current_target and not label:
            label = 'raster'

        # If desired, create new CompoundScan group in HDF5 file, which automatically also creates the first Scan group
        if label:
            dbe.req.k7w_new_compound_scan(target, label, 'cal')
            session.fire_noise_diode(label='', announce=False, **session.nd_params)
        else:
            session.fire_noise_diode(announce=False, **session.nd_params)

        # Create start and end positions of each scan, based on scan parameters
        scan_levels = np.arange(-(num_scans // 2), num_scans // 2 + 1)
        scanning_coord = (scan_extent / 2.0) * (-1) ** scan_levels
        stepping_coord = scan_spacing * scan_levels
        # Flip sign of elevation offsets to ensure that the first scan always starts at the top left of target
        scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)
        scan_ends = zip(-scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, scanning_coord)

        # Perform multiple scans across the target
        for scan_index, (start, end) in enumerate(zip(scan_starts, scan_ends)):

            user_logger.info('slewing to start of scan %d' % (scan_index,))
            with session.start_scan('slew'):
                # Move each antenna to the start position of the next scan
                ants.req.scan_asym(start[0], start[1], end[0], end[1], scan_duration, projection)
                ants.req.mode('POINT')
                # Wait until they are all in position (with 5 minute timeout)
                ants.wait('lock', True, 300)
            user_logger.info('start of scan %d reached' % (scan_index,))

            session.fire_noise_diode(announce=False, **session.nd_params)

            user_logger.info('performing scan %d' % (scan_index,))
            with session.start_scan('scan'):
                # Start scanning the antennas
                ants.req.mode('SCAN')
                # Wait until they are all finished scanning (with 5 minute timeout)
                ants.wait('scan_status', 'after', 300)
            user_logger.info('scan %d complete' % (scan_index,))

            session.fire_noise_diode(announce=False, **session.nd_params)

        return True

    def end(self):
        """End the session, which stops data capturing and closes the data file.

        This does not affect the antennas, which continue to perform their
        last action.

        """
        # Create references to allow easy copy-and-pasting from this function
        session, kat, ants, dbe = self, self.kat, self.ants, self.dbe

        # Obtain the name of the file currently being written to
        reply = dbe.req.k7w_get_current_file()
        outfile = reply[1].replace('writing', 'unaugmented') if reply.succeeded else '<unknown file>'

        msg = 'Scans complete, data captured to %s' % (outfile,)
        user_logger.info(msg)
        # The final output file name after augmentation
        session.output_file = os.path.basename(outfile).replace('.unaugmented', '')

        # Stop the DBE data flow (this indirectly stops k7writer via a stop packet, which then closes the HDF5 file)
        dbe.req.capture_stop()
        msg = 'Ended data capturing session with experiment ID %s' % (session.experiment_id,)
        user_logger.info(msg)
        activity_logger.info(msg)
        if kat.has_connected_device('cfg'):
            kat.cfg.req.set_script_param("script-endtime",
                                         time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time())))
            kat.cfg.req.set_script_param("script-status", "ended")

        if session.stow_when_done and ants is not None:
            user_logger.info("Stowing dishes.")
            activity_logger.info("Stowing dishes.")
            ants.req.mode("STOW")

        user_logger.info("==========================")

        activity_logger.info("----- Script ended  %s (%s) Output file %s" % (sys.argv[0], ' '.join(sys.argv[1:]), session.output_file))

class TimeSession(object):
    """Fake CaptureSession object used to estimate the duration of an experiment."""
    def __init__(self, kat, dbe='dbe', **kwargs):
        self.kat = kat
        # If not a device itself, assume dbe is the name of the device
        if not isinstance(dbe, KATDevice):
            try:
                dbe = getattr(kat, dbe)
            except AttributeError:
                raise ValueError("DBE proxy '%s' not found (i.e. no kat.%s exists)" % (dbe, dbe))
        self.dbe = dbe

        # Default settings for session parameters (in case standard_setup is not called)
        self.ants = None
        self.experiment_id = 'interactive'
        self.record_slews = True
        self.stow_when_done = False
        self.nd_params = {'diode' : 'coupler', 'on' : 0., 'off' : 0., 'period' : -1.}
        self.last_nd_firing = 0.
        self.output_file = ''

        self.start_time = time.time()
        self.time = self.start_time
        self.projection = ('ARC', 0., 0.)

        # Usurp time module functions that deal with the passage of real time, and connect them to session time instead
        self._realtime, self._realsleep = time.time, time.sleep
        time.time = lambda: self.time
        def simsleep(seconds):
            self.time += seconds
        time.sleep = simsleep
        self._fake_ants = []

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
        user_logger.info("DBE proxy used = %s" % (dbe.name,))

        activity_logger.info("Timing simulation. ----- Script starting  %s (%s)" % (sys.argv[0], ' '.join(sys.argv[1:])))

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
        for m in range(len(self._fake_ants)):
            antenna = self._fake_ants[m][0]
            az, el = self._azel(target, self.time, antenna)
            self._fake_ants[m] = (antenna, mode, az, max(el, 2.))

    def _slew_to(self, target, mode='POINT', timeout=300.):
        """Slew antennas to target (or nearest point on horizon), with timeout."""
        slew_times = []
        for ant, ant_mode, ant_az, ant_el in self._fake_ants:
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

    def standard_setup(self, ants, observer, description, experiment_id=None, centre_freq=None,
                       dump_rate=1.0, nd_params=None, record_slews=None, stow_when_done=None, **kwargs):
        """Perform basic experimental setup including antennas, LO and dump rate."""
        self.ants = ant_array(self.kat, ants)
        for ant in self.ants.devs:
            try:
                self._fake_ants.append((katpoint.Antenna(ant.sensor.observer.get_value()),
                                        ant.sensor.mode.get_value(),
                                        ant.sensor.pos_actual_scan_azim.get_value(),
                                        ant.sensor.pos_actual_scan_elev.get_value()))
            except AttributeError:
                pass
        # Override provided session parameters (or initialize them from existing parameters if not provided)
        self.experiment_id = experiment_id = self.experiment_id if experiment_id is None else experiment_id
        self.nd_params = nd_params = self.nd_params if nd_params is None else nd_params
        self.record_slews = record_slews = self.record_slews if record_slews is None else record_slews
        self.stow_when_done = stow_when_done = self.stow_when_done if stow_when_done is None else stow_when_done

        user_logger.info('Antennas used = %s' % (' '.join([ant[0].name for ant in self._fake_ants]),))
        user_logger.info("Observer = %s" % (observer,))
        user_logger.info("Description ='%s'" % (description,))
        user_logger.info("Experiment ID = %s" % (experiment_id,))
        # There is no way to find out the centre frequency in this fake session...
        if centre_freq is None:
            user_logger.info('RF centre frequency = unknown to simulator, dump rate = %g Hz, keep slews = %s' %
                             (dump_rate, record_slews))
        else:
            user_logger.info("RF centre frequency = %g MHz, dump rate = %g Hz, keep slews = %s" %
                             (centre_freq, dump_rate, record_slews))
        if nd_params['period'] > 0:
            nd_info = "Will switch '%s' noise diode on for %g s and off for %g s, every %g s if possible" % \
                      (nd_params['diode'], nd_params['on'], nd_params['off'], nd_params['period'])
        elif nd_params['period'] == 0:
            nd_info = "Will switch '%s' noise diode on for %g s and off for %g s at every opportunity" % \
                      (nd_params['diode'], nd_params['on'], nd_params['off'])
        else:
            nd_info = "Noise diode will not fire automatically"
        user_logger.info(nd_info + " while performing canned commands")
        user_logger.info('--------------------------')

    def capture_start(self):
        """Start capturing data (ignored in version 1, as start is implicit)."""
        pass

    def label(self, label):
        """Add timestamped label to HDF5 file (ignored in version 1)."""
        pass

    def on_target(self, target):
        """Determine whether antennas are tracking a given target."""
        if not self._fake_ants:
            return False
        for antenna, mode, ant_az, ant_el in self._fake_ants:
            az, el = self._azel(target, self.time, antenna)
            # Checking for lock and checking for target identity considered the same thing
            if (az != ant_az) or (el != ant_el) or (mode != 'POINT'):
                return False
        return True

    def target_visible(self, target, duration=0., timeout=300., horizon=2., operation='scan'):
        """Check whether target is visible for given duration."""
        if not self._fake_ants:
            return False
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        horizon = katpoint.deg2rad(horizon)
        # Include an average time to slew to the target (worst case about 90 seconds, so half that)
        now = self.time + 45.
        average_el, visible_before, visible_after = [], [], []
        for antenna, mode, ant_az, ant_el in self._fake_ants:
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

    def fire_noise_diode(self, diode='pin', on=10.0, off=10.0, period=0.0, label='cal', announce=True):
        """Estimate time taken to fire noise diode."""
        if not self._fake_ants:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
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

    def track(self, target, duration=20.0, drive_strategy='shortest-slew', label='track', announce=True):
        """Estimate time taken to perform track."""
        if not self._fake_ants:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
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

    def scan(self, target, duration=30.0, start=(-3.0, 0.0), end=(3.0, 0.0), index=-1,
             projection=default_proj, drive_strategy='shortest-slew', label='scan', announce=True):
        """Estimate time taken to perform single linear scan."""
        if not self._fake_ants:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
        scan_name = 'scan' if index < 0 else 'scan %d' % (index,)
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        if announce:
            user_logger.info("Initiating %g-second scan across target '%s'" % (duration, target.name))
        if not self.target_visible(target, duration):
            user_logger.warning("Skipping track, as target '%s' will be below horizon" % (target.name,))
            return False
        self.fire_noise_diode(label='', announce=False, **self.nd_params)
        projection = Offset.PROJECTIONS[projection]
        self.projection = (projection, start[0], start[1])
        user_logger.info('slewing to start of %s' % (scan_name,))
        self._slew_to(target, mode='SCAN')
        user_logger.info('start of %s reached' % (scan_name,))
        self.fire_noise_diode(announce=False, **self.nd_params)
        # Assume antennas can keep up with target (and doesn't scan too fast either)
        user_logger.info('performing %s' % (scan_name,))
        self.time += duration + 1.0
        user_logger.info('%s complete' % (scan_name,))
        self.fire_noise_diode(announce=False, **self.nd_params)
        self.projection = (projection, end[0], end[1])
        self._teleport_to(target)
        return True

    def raster_scan(self, target, num_scans=3, scan_duration=30.0, scan_extent=6.0, scan_spacing=0.5,
                    scan_in_azimuth=True, projection=default_proj, drive_strategy='shortest-slew',
                    label='raster', announce=True):
        """Estimate time taken to perform raster scan."""
        if not self._fake_ants:
            raise ValueError('No antennas specified for session - please run session.standard_setup first')
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
        # Create start and end positions of each scan, based on scan parameters
        scan_levels = np.arange(-(num_scans // 2), num_scans // 2 + 1)
        scanning_coord = (scan_extent / 2.0) * (-1) ** scan_levels
        stepping_coord = scan_spacing * scan_levels
        # Flip sign of elevation offsets to ensure that the first scan always starts at the top left of target
        scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)
        scan_ends = zip(-scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, scanning_coord)
        self.fire_noise_diode(label='', announce=False, **self.nd_params)
        # Perform multiple scans across the target
        for scan_index, (start, end) in enumerate(zip(scan_starts, scan_ends)):
            self.projection = (projection, start[0], start[1])
            user_logger.info('slewing to start of scan %d' % (scan_index,))
            self._slew_to(target, mode='SCAN')
            user_logger.info('start of scan %d reached' % (scan_index,))
            self.fire_noise_diode(announce=False, **self.nd_params)
            # Assume antennas can keep up with target (and doesn't scan too fast either)
            user_logger.info('performing scan %d' % (scan_index,))
            self.time += scan_duration + 1.0
            user_logger.info('scan %d complete' % (scan_index,))
            self.fire_noise_diode(announce=False, **self.nd_params)
            self.projection = (projection, end[0], end[1])
            self._teleport_to(target)
        return True

    def end(self):
        """Stop data capturing to shut down the session and close the data file."""
        user_logger.info('Scans complete, no data captured as this is a timing simulation...')
        msg = 'Ended data capturing session with experiment ID %s' % (self.experiment_id,)
        user_logger.info(msg)
        activity_logger.info("Timing simulation. %s" % (msg,))
        if self.stow_when_done and self._fake_ants:
            msg = "Stowing dishes."
            user_logger.info(msg)
            activity_logger.info("Timing simulation.  %s" % (msg,))
            self._teleport_to(katpoint.Target("azel, 0.0, 90.0"), mode="STOW")
        user_logger.info('==========================')
        duration = self.time - self.start_time
        if duration <= 100:
            duration = '%d seconds' % (np.ceil(duration),)
        elif duration <= 100 * 60:
            duration = '%d minutes' % (np.ceil(duration / 60.),)
        else:
            duration = '%.1f hours' % (duration / 3600.,)
        msg = "Experiment estimated to last %s until this time" % (duration,)
        user_logger.info(msg+"\n")
        activity_logger.info("Timing simulation.  %s" % (msg,))
        # Restore time module functions
        time.time, time.sleep = self._realtime, self._realsleep
        # Restore logging
        for handler in user_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.formatter.datefmt = handler.formatter.old_datefmt
                del handler.formatter.old_datefmt
            else:
                handler.setLevel(handler.old_level)
                del handler.old_level

        activity_logger.info("Timing simulation. ----- Script ended  %s (%s)" % (sys.argv[0], ' '.join(sys.argv[1:])))
