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

from .array import Array
from .katcp_client import KATDevice
from .defaults import user_logger
from .utility import tbuild

# Ripped from katpoint.construct_target_params, to avoid extra dependencies
def preferred_name(description):
    """Parse target description string to extract preferred target name."""
    fields = [s.strip() for s in description.split(',')]
    # Extract preferred name from name list (starred or first entry)
    names = [s.strip() for s in fields[0].split('|')]
    if len(names) == 0:
        return ''
    first_word = names[0].split(' ')[0]
    if first_word in ('azel', 'radec'):
        return first_word
    if first_word == 'xephem':
        edb_string = fields[-1].replace('~', ',')
        edb_name_field = edb_string.partition(',')[0]
        edb_names = [name.strip() for name in edb_name_field.split('|')]
        return edb_names[0]
    try:
        ind = [name.startswith('*') for name in names].index(True)
        return names[ind][1:]
    except ValueError:
        return names[0]

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
    record_slews : {True, False}
        If True, correlator data is recorded contiguously and the data file
        includes 'slew' scans which occur while the antennas slew to the start
        of the next proper scan. If False, the file output (but not the signal
        displays) is paused while the antennas slew, and the data file contains
        only proper scans.

    """
    def __init__(self, kat, label, record_slews):
        self.kat = kat
        self.label = label
        self.record_slews = record_slews

    def __enter__(self):
        """Start with scan and start/unpause capturing if necessary."""
        # Do nothing if we want to slew and slews are not to be recorded
        if self.label == 'slew' and not self.record_slews:
            return self
        # Create new Scan group in HDF5 file, if label is non-empty
        if self.label:
            self.kat.dbe.req.k7w_new_scan(self.label)
        # Unpause HDF5 file output (redundant if output is never paused anyway)
        self.kat.dbe.req.k7w_write_hdf5(1)
        # If we haven't yet, start recording data from the correlator (which creates the data file)
        if self.kat.dbe.sensor.capturing.get_value() == '0':
            self.kat.dbe.req.capture_start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finish with scan and pause capturing if necessary."""
        # If slews are not to be recorded, pause the file output again afterwards
        if not self.record_slews:
            self.kat.dbe.req.k7w_write_hdf5(0)
        # Do not suppress any exceptions that occurred in the body of with-statement
        return False

class CaptureSession(object):
    """Context manager that encapsulates a single data capturing session.

    A data capturing *session* results in a single data file, potentially
    containing multiple scans and compound scans. An *experiment* may consist of
    multiple sessions. This object ensures that the capturing process is
    started and completed cleanly, even if exceptions occur during the session.
    It also provides canned routines for simple observations such as tracks,
    single scans, raster scans and holography scans on a specific source.

    The initialisation of the session object selects a sub-array of antennas,
    prepares the data capturing subsystem (DBE and k7writer) and sets the RFE7
    LO frequency.

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
    centre_freq : float, optional
        RF centre frequency, in MHz
    dump_rate : float, optional
        Correlator dump rate, in Hz
    record_slews : {True, False}, optional
        If True, correlator data is recorded contiguously and the data file
        includes 'slew' scans which occur while the antennas slew to the start
        of the next proper scan. If False, the file output (but not the signal
        displays) is paused while the antennas slew, and the data file contains
        only proper scans.
    nd_params : dict, optional
        Dictionary containing parameters that control firing of the noise diode.
        These parameters are in the form of keyword-value pairs, and matches the
        parameters of the :meth:`fire_noise_diode` method.
    kwargs : dict
        Ignore any other keyword arguments (simplifies passing options as dict)

    Raises
    ------
    ValueError
        If antenna with a specified name is not found on KAT connection object

    """
    def __init__(self, kat, experiment_id, observer, description, ants,
                 centre_freq=1800.0, dump_rate=1.0, record_slews=True,
                 nd_params={'diode' : 'pin', 'on_duration' : 10.0,
                            'off_duration' : 10.0, 'period' : 180.}, **kwargs):
        try:
            self.kat = kat

            user_logger.info("New data capturing session")
            # Log the activity parameters (if config manager is around)
            if kat.has_connected_device('cfg'):
                kat.cfg.req.set_script_param("script-session-status", "initialising")
                kat.cfg.req.set_script_param("script-starttime", "")
                kat.cfg.req.set_script_param("script-endtime", "")
                kat.cfg.req.set_script_param("script-name", sys.argv[0])
                kat.cfg.req.set_script_param("script-arguments", ' '.join(sys.argv[1:]))
                kat.cfg.req.set_script_param("script-experiment-id", experiment_id)
                kat.cfg.req.set_script_param("script-observer", observer)
                kat.cfg.req.set_script_param("script-description", description)
                kat.cfg.req.set_script_param("script-rf-params", "Freq=%g MHz, Dump rate=%g Hz, Keep slews=%s" %
                                                                 (centre_freq, dump_rate, record_slews))
            user_logger.info("--------------------------")
            user_logger.info("Experiment ID = %s" % (experiment_id,))
            user_logger.info("Observer = %s" % (observer,))
            user_logger.info("Description ='%s'" % description)
            user_logger.info("RF centre frequency = %g MHz, dump rate = %g Hz, keep slews = %s" %
                             (centre_freq, dump_rate, record_slews))

            self.ants = ants = ant_array(kat, ants)
            self.experiment_id = experiment_id
            self.record_slews = record_slews
            self.nd_params = nd_params
            self.last_nd_firing = 0.

            # Start with a clean state, by stopping the DBE
            kat.dbe.req.capture_stop()

            # Set centre frequency in RFE stage 7
            kat.rfe7.req.rfe7_lo1_frequency(4200.0 + centre_freq, 'MHz')
            effective_lo_freq = (centre_freq - 200.0) * 1e6

            # Set data output directory (typically on ff-dc machine)
            kat.dbe.req.k7w_output_directory("/var/kat/data")
            # Enable output to HDF5 file (takes effect on capture_start only), and set basic experimental info
            kat.dbe.req.k7w_write_hdf5(1)
            kat.dbe.req.k7w_experiment_info(experiment_id, observer, description)

            # The DBE proxy needs to know the dump period (in ms) as well as the effective LO freq,
            # which is used for fringe stopping (eventually). This sets the delay model and other
            # correlator parameters, such as the dump rate, and instructs the correlator to pass
            # its data to the k7writer daemon (set via configuration)
            kat.dbe.req.capture_setup(1000.0 / dump_rate, effective_lo_freq)

             # If the DBE is simulated, it will have position update commands
            if hasattr(kat.dbe.req, 'dbe_pointing_az') and hasattr(kat.dbe.req, 'dbe_pointing_el'):
                first_ant = ants.devs[0]
                # Tell the DBE simulator where the first antenna is so that it can generate target flux at the right time
                # The minimum time between position updates is just a little less than the standard (az, el) sensor period
                first_ant.sensor.pos_actual_scan_azim.register_listener(kat.dbe.req.dbe_pointing_az, 0.4)
                first_ant.sensor.pos_actual_scan_elev.register_listener(kat.dbe.req.dbe_pointing_el, 0.4)
                user_logger.info("DBE simulator receives position updates from antenna '%s'" % (first_ant.name,))

            if kat.has_connected_device('cfg'):
                kat.cfg.req.set_script_param("script-session-status", "initialised")

        except Exception, e:
            user_logger.error("CaptureSession failed to initialise (%s)" % (e,))
            raise

    def __enter__(self):
        """Enter the data capturing session."""
        if self.kat.has_connected_device('cfg'):
            self.kat.cfg.req.set_script_param("script-session-status", "running")
            self.kat.cfg.req.set_script_param("script-starttime",
                                              time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time())))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the data capturing session, closing the data file."""
        if self.kat.has_connected_device('cfg'):
            self.kat.cfg.req.set_script_param("script-session-status", "exiting")
            self.kat.cfg.req.set_script_param("script-endtime",
                                              time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time())))
            if exc_value is not None:
                user_logger.error('Session interrupted by exception (%s)' % (exc_value,))
        self.shutdown()
        # Do not suppress any exceptions that occurred in the body of with-statement
        return False

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
                print ant.name, ant.sensor.target.get_value(), ant.sensor.mode.get_value(), ant.sensor.lock.get_value()
                return False
        return True

    def target_visible(self, target, duration=0., timeout=300., horizon=2., operation='scan'):
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
        operation : string, optional
            Description of current observation, for use in warning message

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
            user_logger.warning("Target '%s' is never visible during %s (average elevation is %g degrees)" %
                                (target.name, operation, np.mean(average_el)))
        else:
            user_logger.warning("Target '%s' will rise or set during %s" % (target.name, operation))
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
        return CaptureScan(self.kat, label, self.record_slews)

    def fire_noise_diode(self, diode='pin', on_duration=10.0, off_duration=10.0, period=0.0, label='cal'):
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
        The specified *off_duration* is therefore a minimum value. Remember to
        run :meth:`shutdown` to close the file and finally stop the observation
        (automatically done when this object is used in a with-statement)!

        Parameters
        ----------
        diode : {'pin', 'coupler'}
            Noise diode source to use (pin diode is situated in feed horn and
            produces high-level signal, while coupler diode couples into
            electronics after the feed at a much lower level)
        on_duration : float, optional
            Minimum duration for which diode is switched on, in seconds
        off_duration : float, optional
            Minimum duration for which diode is switched off, in seconds
        period : float, optional
            Minimum time between noise diode firings, in seconds. (The maximum
            time is determined by the duration of individual slews and scans,
            which are considered atomic and won't be interrupted.) If 0, fire
            diode unconditionally. If negative, don't fire diode at all.
        label : string
            Label for scan in HDF5 file, usually single computer-parseable word.
            If this is an empty string, do not create new Scan group in the file.

        Returns
        -------
        fired : {True, False}
            True if noise diode fired

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # If period is non-negative, quit if it is not yet time to fire the noise diode
        if period < 0.0 or (time.time() - session.last_nd_firing) < period:
            return False
        # Find pedestal controllers with the same number as antennas (i.e. 'ant1' maps to 'ped1') and put into Array
        pedestals = Array('peds', [getattr(kat, 'ped' + ant.name[3:]) for ant in ants.devs])

        user_logger.info("Firing '%s' noise diode (%g seconds on, %g seconds off)" % (diode, on_duration, off_duration))

        with session.start_scan(label):
            # Switch noise diode on on all antennas
            pedestals.req.rfe3_rfe15_noise_source_on(diode, 1, 'now', 0)
            time.sleep(on_duration)
            # Mark on -> off transition as last firing
            session.last_nd_firing = time.time()
            # Switch noise diode off on all antennas
            pedestals.req.rfe3_rfe15_noise_source_on(diode, 0, 'now', 0)
            time.sleep(off_duration)
        return True

    def track(self, target, duration=20.0, drive_strategy='longest-track', label='track'):
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
        drive_strategy : {'longest-track', 'shortest-slew'}, optional
            Drive strategy employed by antennas, used to decide what to do when
            target is in azimuth overlap region of antenna. The default is to
            go to the wrap that will permit the longest possible track before
            the target sets.
        label : string, optional
            Label for compound scan in HDF5 file, usually a single word. If this
            is an empty string and *target* matches the target of the current
            compound scan being written, do not create new CompoundScan group
            in the file.

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # Turn target object into description string (or use string as is)
        target = getattr(target, 'description', target)
        # Check if we are currently on the desired target and whether the target will be visible
        on_target = self.on_target(target)
        self.target_visible(target, duration, operation='track')

        # Set the drive strategy for how antenna moves between targets
        ants.req.drive_strategy(drive_strategy)
        # Set the antenna target (antennas will already move there if in mode 'POINT')
        ants.req.target(target)
        # Provide target to the DBE proxy, which will use it as delay-tracking center
        kat.dbe.req.target(target)
        # Obtain target associated with the current compound scan
        req = kat.dbe.req.k7w_get_target()
        current_target = req[1] if req else ''
        # Ensure that there is a label if a new compound scan is forced
        if target != current_target and not label:
            label = 'track'

        # If desired, create new CompoundScan group in HDF5 file, which automatically also creates the first Scan group
        if label:
            kat.dbe.req.k7w_new_compound_scan(target, label, 'cal')
            session.fire_noise_diode(label='', **session.nd_params)
        else:
            session.fire_noise_diode(**session.nd_params)

        # Avoid slewing if we are already on target
        if not on_target:
            user_logger.info("Slewing to target '%s'" % (preferred_name(target),))
            with session.start_scan('slew'):
                # Start moving each antenna to the target
                ants.req.mode('POINT')
                # Wait until they are all in position (with 5 minute timeout)
                ants.wait('lock', True, 300)

            session.fire_noise_diode(**session.nd_params)

        user_logger.info("Tracking target '%s' for %g seconds" % (preferred_name(target), duration))
        with session.start_scan('scan'):
            # Do nothing else for the duration of the track
            time.sleep(duration)

        session.fire_noise_diode(**session.nd_params)

    def scan(self, target, duration=30.0, start=-3.0, end=3.0, scan_in_azimuth=True,
             drive_strategy='shortest-slew', label='scan'):
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
        start : float, optional
            Start offset of scan along scanning coordinate, in degrees
            (see *Notes* below)
        end : float, optional
            End offset of scan along scanning coordinate, in degrees
            (see *Notes* below)
        scan_in_azimuth : {True, False}, optional
            True if azimuth changes during scan while elevation remains fixed;
            False if scanning in elevation and stepping in azimuth instead
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

        Notes
        -----
        Take note that scanning is done in a projection on the celestial sphere,
        and the scan start and end are in the projected coordinates. The azimuth
        coordinate of a scan in azimuth will therefore change more than the
        *start* and *end* parameters suggest, especially at high elevations.
        This ensures that the same scan parameters will lead to the same
        qualitative scan for any position on the celestial sphere.

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # Turn target object into description string (or use string as is)
        target = getattr(target, 'description', target)
        # Check whether the target will be visible
        self.target_visible(target, duration, operation='scan')

        # Set the drive strategy for how antenna moves between targets
        ants.req.drive_strategy(drive_strategy)
        # Set the antenna target
        ants.req.target(target)
        # Provide target to the DBE proxy, which will use it as delay-tracking center
        kat.dbe.req.target(target)
        # Obtain target associated with the current compound scan
        req = kat.dbe.req.k7w_get_target()
        current_target = req[1] if req else ''
        # Ensure that there is a label if a new compound scan is forced
        if target != current_target and not label:
            label = 'scan'

        # If desired, create new CompoundScan group in HDF5 file, which automatically also creates the first Scan group
        if label:
            kat.dbe.req.k7w_new_compound_scan(target, label, 'cal')
            session.fire_noise_diode(label='', **session.nd_params)
        else:
            session.fire_noise_diode(**session.nd_params)

        user_logger.info("Slewing to start of scan across target '%s'" % (preferred_name(target),))
        with session.start_scan('slew'):
            # Move each antenna to the start position of the scan
            if scan_in_azimuth:
                ants.req.scan_asym(start, 0.0, end, 0.0, duration)
            else:
                ants.req.scan_asym(0.0, start, 0.0, end, duration)
            ants.req.mode('POINT')
            # Wait until they are all in position (with 5 minute timeout)
            ants.wait('lock', True, 300)

        session.fire_noise_diode(**session.nd_params)

        user_logger.info("Scanning across target '%s' for %g seconds" % (preferred_name(target), duration))
        with session.start_scan('scan'):
            # Start scanning the antennas
            ants.req.mode('SCAN')
            # Wait until they are all finished scanning (with 5 minute timeout)
            ants.wait('scan_status', 'after', 300)

        session.fire_noise_diode(**session.nd_params)

    def raster_scan(self, target, num_scans=3, scan_duration=30.0,
                    scan_extent=6.0, scan_spacing=0.5, scan_in_azimuth=True,
                    drive_strategy='shortest-slew', label='raster'):
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

        Notes
        -----
        Take note that scanning is done in a projection on the celestial sphere,
        and the scan extent and spacing apply to the projected coordinates.
        The azimuth coordinate of a scan in azimuth will therefore change more
        than the *scan_extent* parameter suggests, especially at high elevations.
        This ensures that the same scan parameters will lead to the same
        qualitative scan for any position on the celestial sphere.

        """
        # Create reference to session, KAT object and antennas, as this allows easy copy-and-pasting from this function
        session, kat, ants = self, self.kat, self.ants
        # Turn target object into description string (or use string as is)
        target = getattr(target, 'description', target)
        # Calculate average time that noise diode is operated per scan, to add to scan duration in check below
        nd_time = session.nd_params['on_duration'] + session.nd_params['off_duration']
        nd_time /= (max(session.nd_params['period'], scan_duration) / scan_duration)
        # Check whether the target will be visible for entire duration of raster scan
        self.target_visible(target, (scan_duration + nd_time) * num_scans, operation='raster scan')

        # Set the drive strategy for how antenna moves between targets
        ants.req.drive_strategy(drive_strategy)
        # Set the antenna target
        ants.req.target(target)
        # Provide target to the DBE proxy, which will use it as delay-tracking center
        kat.dbe.req.target(target)
        # Obtain target associated with the current compound scan
        req = kat.dbe.req.k7w_get_target()
        current_target = req[1] if req else ''
        # Ensure that there is a label if a new compound scan is forced
        if target != current_target and not label:
            label = 'raster'

        # If desired, create new CompoundScan group in HDF5 file, which automatically also creates the first Scan group
        if label:
            kat.dbe.req.k7w_new_compound_scan(target, label, 'cal')
            session.fire_noise_diode(label='', **session.nd_params)
        else:
            session.fire_noise_diode(**session.nd_params)

        # Create start positions of each scan, based on scan parameters
        scan_steps = np.arange(-(num_scans // 2), num_scans // 2 + 1)
        scanning_coord = (scan_extent / 2.0) * (-1) ** scan_steps
        stepping_coord = scan_spacing * scan_steps
        # These minus signs ensure that the first scan always starts at the top left of target
        scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)

        # Iterate through the scans across the target
        for scan_count, scan in enumerate(scan_starts):

            user_logger.info("Slewing to start of scan %d of %d across target '%s'" %
                             (scan_count + 1, len(scan_starts), preferred_name(target)))
            with session.start_scan('slew'):
                # Move each antenna to the start position of the next scan
                if scan_in_azimuth:
                    ants.req.scan_asym(scan[0], scan[1], -scan[0], scan[1], scan_duration)
                else:
                    ants.req.scan_asym(scan[0], scan[1], scan[0], -scan[1], scan_duration)
                ants.req.mode('POINT')
                # Wait until they are all in position (with 5 minute timeout)
                ants.wait('lock', True, 300)

            session.fire_noise_diode(**session.nd_params)

            user_logger.info("Performing scan %d of %d across target '%s' for %g seconds" %
                             (scan_count + 1, len(scan_starts), preferred_name(target), scan_duration))
            with session.start_scan('scan'):
                # Start scanning the antennas
                ants.req.mode('SCAN')
                # Wait until they are all finished scanning (with 5 minute timeout)
                ants.wait('scan_status', 'after', 300)

            session.fire_noise_diode(**session.nd_params)

    def shutdown(self):
        """Stop data capturing to shut down the session and close the data file.

        This does not affect the antennas, which continue to perform their
        last action.

        """
        # Create reference to session and KAT objects, as this allows easy copy-and-pasting from this function
        session, kat = self, self.kat
        # Obtain the name of the file currently being written to
        reply = kat.dbe.req.k7w_get_current_file()
        outfile = reply[1].replace('writing', 'unaugmented') if reply.succeeded else '<unknown file>'
        user_logger.info('Scans complete, data captured to %s' % (outfile,))

        # Stop the DBE data flow (this indirectly stops k7writer via a stop packet, which then closes the HDF5 file)
        kat.dbe.req.capture_stop()
        user_logger.info('Ended data capturing session with experiment ID %s' % (session.experiment_id,))
        if kat.has_connected_device('cfg'):
            kat.cfg.req.set_script_param("script-session-status", "done")


class TimeSession(object):
    """Fake CaptureSession object used to estimate the duration of an experiment."""
    def __init__(self, kat, experiment_id, observer, description, ants,
                 centre_freq=1800.0, dump_rate=1.0, record_slews=True,
                 nd_params={'diode' : 'pin', 'on_duration' : 10.0,
                            'off_duration' : 10.0, 'period' : 180.}, **kwargs):
        self.start_time = time.time()
        self.time = self.start_time
        self.ants = []
        for ant in ant_array(kat, ants).devs:
            try:
                self.ants.append((katpoint.Antenna(ant.sensor.observer.get_value()),
                                  ant.sensor.mode.get_value(),
                                  ant.sensor.pos_actual_scan_azim.get_value(),
                                  ant.sensor.pos_actual_scan_elev.get_value()))
            except AttributeError:
                pass
        self.nd_params = nd_params
        self.last_nd_firing = 0.
        self.projection = ('ARC', 0., 0.)
        self.realtime = None
        self.realsleep = None
        print "\nEstimating duration of experiment starting %s (nothing real will happen!)" % (self._time_str(),)
        print "~~~~~~~~~~"
        print "~ %s INFO     Experiment ID = %s" % (self._time_str(), experiment_id,)
        print "~ %s INFO     Observer = %s" % (self._time_str(), observer,)
        print "~ %s INFO     Description ='%s'" % (self._time_str(), description)
        print "~ %s INFO     RF centre frequency = %g MHz, dump rate = %g Hz, keep slews = %s" % \
              (self._time_str(), centre_freq, dump_rate, record_slews)

    def __enter__(self):
        """Start time estimate, overriding the time module."""
        # Usurp time module functions that deal with the passage of real time, and connect them to session time instead
        time = sys.modules['time']
        self.realtime, self.realsleep = time.time, time.sleep
        time.time = lambda: self.time
        def simsleep(seconds):
            self.time += seconds
        time.sleep = simsleep
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finish time estimate, restoring the time module."""
        duration = self.time - self.start_time
        if duration <= 100:
            duration = '%d seconds' % (np.ceil(duration),)
        elif duration <= 100 * 60:
            duration = '%d minutes' % (np.ceil(duration / 60.),)
        else:
            duration = '%.1f hours' % (duration / 3600.,)
        print "~~~~~~~~~~"
        print "Experiment estimated to last %s until %s\n" % (duration, self._time_str())
        # Restore time module functions
        time = sys.modules['time']
        time.time, time.sleep = self.realtime, self.realsleep
        # Do not suppress any exceptions that occurred in the body of with-statement
        return False

    def _time_str(self):
        """Current session timestamp (in local timezone) as a string."""
        return time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(self.time))

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
        self.time += np.max(slew_times)
        # Blindly assume all antennas are on target (or on horizon) after this interval
        self._teleport_to(target, mode)

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
            print "~ %s WARNING  Target '%s' is never visible during %s (average elevation is %g degrees)" % \
                  (self._time_str(), target.name, operation, np.mean(average_el))
        else:
            print "~ %s WARNING  Target '%s' will rise or set during %s" % (self._time_str(), target.name, operation)
        return False

    def start_scan(self, label):
        """Starting scan has no major timing effect."""
        pass

    def fire_noise_diode(self, diode='pin', on_duration=10.0, off_duration=10.0, period=0.0, label='cal'):
        """Estimate time taken to fire noise diode."""
        if period < 0.0 or (self.time - self.last_nd_firing) < period:
            return False
        print "~ %s INFO     Firing '%s' noise diode (on %g seconds, off %g seconds)" % \
              (self._time_str(), diode, on_duration, off_duration)
        self.time += on_duration
        self.last_nd_firing = self.time + 0.
        self.time += off_duration
        return True

    def track(self, target, duration=20.0, drive_strategy='longest-track', label='track'):
        """Estimate time taken to perform track."""
        target = target if hasattr(target, 'description') else katpoint.Target(target)
        self.target_visible(target, duration, operation='track')
        self.fire_noise_diode(label='', **self.nd_params)
        if not self.on_target(target):
            print "~ %s INFO     Slewing to target '%s'" % (self._time_str(), target.name,)
            self._slew_to(target)
            self.fire_noise_diode(**self.nd_params)
        print "~ %s INFO     Tracking target '%s'" % (self._time_str(), target.name,)
        self.time += duration + 1.0
        self.fire_noise_diode(**self.nd_params)
        self._teleport_to(target)

    def scan(self, target, duration=30.0, start=-3.0, end=3.0, scan_in_azimuth=True,
             drive_strategy='shortest-slew', label='scan'):
        """Estimate time taken to perform single linear scan."""
        target = target if hasattr(target, 'description') else katpoint.Target(target)
        self.target_visible(target, duration, operation='scan')
        self.fire_noise_diode(label='', **self.nd_params)
        print "~ %s INFO     Slewing to start of scan across target '%s'" % (self._time_str(), target.name,)
        self.projection = ('ARC', start, 0.) if scan_in_azimuth else ('ARC', 0., start)
        self._slew_to(target, mode='SCAN')
        self.fire_noise_diode(**self.nd_params)
        print "~ %s INFO     Starting scan across target '%s'" % (self._time_str(), target.name,)
        # Assume antennas can keep up with target (and doesn't scan too fast either)
        self.time += duration + 1.0
        self.fire_noise_diode(**self.nd_params)
        self.projection = ('ARC', end, 0.) if scan_in_azimuth else ('ARC', 0., end)
        self._teleport_to(target)

    def raster_scan(self, target, num_scans=3, scan_duration=30.0,
                    scan_extent=6.0, scan_spacing=0.5, scan_in_azimuth=True,
                    drive_strategy='shortest-slew', label='raster'):
        """Estimate time taken to perform raster scan."""
        target = target if hasattr(target, 'description') else katpoint.Target(target)
        nd_time = self.nd_params['on_duration'] + self.nd_params['off_duration']
        nd_time /= (max(self.nd_params['period'], scan_duration) / scan_duration)
        self.target_visible(target, (scan_duration + nd_time) * num_scans, operation='raster scan')
        # Create start positions of each scan, based on scan parameters
        scan_steps = np.arange(-(num_scans // 2), num_scans // 2 + 1)
        scanning_coord = (scan_extent / 2.0) * (-1) ** scan_steps
        stepping_coord = scan_spacing * scan_steps
        # These minus signs ensure that the first scan always starts at the top left of target
        scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)
        self.fire_noise_diode(label='', **self.nd_params)
        # Iterate through the scans across the target
        for scan_count, scan in enumerate(scan_starts):
            print "~ %s INFO     Slewing to start of scan %d of %d on target '%s'" % \
                  (self._time_str(), scan_count + 1, len(scan_starts), target.name)
            self.projection = ('ARC', scan[0], scan[1])
            self._slew_to(target, mode='SCAN')
            self.fire_noise_diode(**self.nd_params)
            print "~ %s INFO     Starting scan %d of %d on target '%s'" % \
                  (self._time_str(), scan_count + 1, len(scan_starts), target.name)
            # Assume antennas can keep up with target (and doesn't scan too fast either)
            self.time += scan_duration + 1.0
            self.fire_noise_diode(**self.nd_params)
            self.projection = ('ARC', -scan[0], scan[1]) if scan_in_azimuth else ('ARC', scan[0], -scan[1])
            self._teleport_to(target)


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

    parser.add_option('-i', '--ini_file', help='System configuration file to use, relative to conf directory ' +
                      '(default reuses existing connection, or falls back to systems/local.conf)')
    parser.add_option('-u', '--experiment_id', help='Experiment ID used to link various parts of experiment ' +
                      'together (UUID generated by default)')
    parser.add_option('-o', '--observer', help='Name of person doing the observation (**required**)')
    parser.add_option('-d', '--description', default='No description.',
                      help='Description of observation (default="%default")')
    parser.add_option('-a', '--ants', metavar='ANTS', help="Comma-separated list of antennas to include " +
                      "(e.g. 'ant1,ant2'), or 'all' for all antennas (**required** - safety reasons)")
    parser.add_option('-f', '--centre_freq', type='float', default=1822.0,
                      help='Centre frequency, in MHz (default="%default")')
    parser.add_option('-r', '--dump_rate', type="float", default=1.0, help='Dump rate, in Hz (default="%default")')
    parser.add_option('-w', '--discard_slews', dest='record_slews', action="store_false", default=True,
                      help='Do not record all the time, i.e. pause while antennas are slewing to the next target')
    parser.add_option('-n', '--nd_params', default='pin,10,10,180',
                      help='Noise diode parameters as "diode,on,off,period", in seconds (default="%default")')

    return parser

def verify_and_connect(opts):
    """Verify command-line options, build KAT configuration and connect to devices.

    Parameters
    ----------
    opts : :class:`optparse.Values` object
        Parsed command-line options (will be updated by this function)

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
    if opts.ants is None:
        raise ValueError('Please specify the antennas to use via -a option (yes, this is a non-optional option...)')
    if opts.observer is None:
        raise ValueError('Please specify the observer name via -o option (yes, this is a non-optional option...)')
    if opts.experiment_id is None:
        # Generate unique string via RFC 4122 version 1
        opts.experiment_id = str(uuid.uuid1())

    # Verify noise diode parameters (should be 'string,number,number,number') and convert to dict
    try:
        opts.nd_params = eval("{'diode':'%s', 'on_duration':%s, 'off_duration':%s, 'period':%s}" %
                              tuple(opts.nd_params.split(',')), {})
    except TypeError, NameError:
        raise ValueError("Noise diode parameters are incorrect (should be 'diode,on,off,period')")
    for key in ('on_duration', 'off_duration', 'period'):
        if opts.nd_params[key] != float(opts.nd_params[key]):
            raise ValueError("Parameter nd_params['%s'] = %s (should be a number)" % (key, opts.nd_params[key]))

    # Try to build the given KAT configuration (which might be None, in which case try to reuse latest active connection)
    # This connects to all the proxies and devices and queries their commands and sensors
    try:
        kat = tbuild(opts.ini_file)
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
