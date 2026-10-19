#!/usr/bin/env python
# Observation script for online VLBI observations. 

import hashlib
import io
from pathlib import Path
import time
import numpy as np
import scipy.ndimage
import katpoint
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                start_session, user_logger, SessionCBF,
                                CalSolutionsUnavailable)


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track one or more sources for a specified time. At least one ' \
              'target must be specified. Note also some **required** options below.'
parser = standard_script_options(usage=usage, description=description)
# Add experiment-specific options
parser.add_option('--max-wait-time', type='float', default=1200.0,
                  help="Maximum time telescope will wait for target UTC "
                       "observation window rather than terminating observation, "
                       "in seconds (default=%default)")
parser.add_option('--min-scan-duration', type='float', default=10.0,
                  help="Minimum accommodated scan duration. Scan will be  "
                       "skipped if possible track time is shorter than this "
                       "value, in seconds (default=%default)")
parser.add_option('--max-mpcc-delay', type='float', default=60.0,
                  help='Max tollerated delay in start time of phase correction '
                       'operation, in seconds. (default=%default)')
parser.add_option('--bf-gain', type=float, default=0.5,
                  help='Set beamformer quantisation gain (default=%default/sqrt(N_ants))')
parser.add_option('--bf-inst', type='string', default='narrow1',
                  help='Instrument on which to setup beamformer weights and gains.'
                       '(default=%default)')
parser.add_option('--bf-core-only', action="store_true", default=False,
                  help='Zero beamformer weights on all but core antennas.')
parser.add_option('--reset-gains', action="store_true", default=False,
                  help='Reset gains prior to phase correction.')
parser.add_option('--skip-wide-correction', action="store_true", default=False,
                  help='Skip requantisation gain updates on wide instrument.')
parser.add_option('--max-gap-MHz', type='float', default=256.0,
                  help='The maximum gap in the bandpass gain that will still be '
                       'interpolated across, in MHz (default=%default)')
parser.add_option('--nb-center-freq', type='float', default=1630.0e6,
                  help='Center frequency of the narrowband instrument.')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='VLBI Observation Test',
                    nd_params='off', quorum=0.9)
# Parse the command line
opts, args = parser.parse_args()
if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")


# Defined functions / parameters
core_ants = ['m000', 'm001', 'm002', 'm003', 'm004', 'm005', 'm006', 'm007', 
             'm008', 'm009', 'm010', 'm011', 'm012', 'm013', 'm014', 'm015', 
             'm016', 'm017', 'm018', 'm019', 'm020', 'm021', 'm022', 'm023', 
             'm024', 'm025', 'm026', 'm027', 'm028', 'm029', 'm030', 'm031', 
             'm034', 'm035', 'm036', 'm037', 'm038', 'm039', 'm040', 'm041', 
             'm042', 'm043', 'm047']

# Instrument default f-engine gains for l-band
rqgain_narrow = 5200.0
rqgain_wide = 3234.0
    

def clean_bandpass(bp_gains, cal_channel_freqs, max_gap_Hz):
    """Clean up bandpass gains by linear interpolation across narrow flagged regions."""
    clean_gains = {}
    # Linearly interpolate across flagged regions as long as they are not too large
    for inp, bp in bp_gains.items():
        flagged = np.isnan(bp)
        if flagged.all():
            clean_gains[inp] = bp
            continue
        chans = np.arange(len(bp))
        interp_bp = np.interp(chans, chans[~flagged], bp[~flagged])
        # Identify flagged regions and tag each with unique integer label
        gaps, n_gaps = scipy.ndimage.label(flagged)
        for n in range(n_gaps):
            gap = np.nonzero(gaps == n + 1)[0]
            gap_freqs = cal_channel_freqs[gap]
            lower = gap_freqs.min()
            upper = gap_freqs.max()
            if upper - lower > max_gap_Hz:
                interp_bp[gap] = np.nan
        clean_gains[inp] = interp_bp
    return clean_gains


def calculate_corrections(G_gains, B_gains, delays, cal_channel_freqs,
                          target_average_correction):
    """Turn cal pipeline products into corrections to be passed to F-engine."""
    average_gain = {}
    gain_corrections = {}
    # First find relative corrections per input with arbitrary global average
    for inp in G_gains:
        # Combine all calibration products for input into single array of gains
        K_gains = np.exp(-2j * np.pi * delays[inp] * cal_channel_freqs)
        gains = K_gains * B_gains[inp] * G_gains[inp]
        if np.isnan(gains).all():
            average_gain[inp] = gain_corrections[inp] = 0.0
            continue
        abs_gains = np.abs(gains)
        # Track the average gain to fix overall power level (and as diagnostic)
        average_gain[inp] = np.nanmedian(abs_gains)
        corrections = 1.0 / gains
        gain_corrections[inp] = np.nan_to_num(corrections)
    # All invalid gains (NaNs) have now been turned into zeros
    valid_average_gains = [g for g in average_gain.values() if g > 0]
    if not valid_average_gains:
        raise ValueError("All gains invalid and beamformer output will be zero!")
    global_average_gain = np.median(valid_average_gains)
    # Iterate over inputs again and fix average values of corrections
    for inp in sorted(G_gains):
        relative_gain = average_gain[inp] / global_average_gain
        if relative_gain == 0.0:
            user_logger.warning("%s has no valid gains and will be zeroed", inp)
            continue
        # This ensures that input at the global average gets target correction
        gain_corrections[inp] *= target_average_correction * global_average_gain
        safe_relative_gain = np.clip(relative_gain, 0.5, 2.0)
        if relative_gain == safe_relative_gain:
            user_logger.info("%s: average gain relative to global average = %5.2f",
                             inp, relative_gain)
        else:
            user_logger.warning("%s: average gain relative to global average "
                                "= %5.2f out of range, clipped to %.1f",
                                inp, relative_gain, safe_relative_gain)
            gain_corrections[inp] *= relative_gain / safe_relative_gain
    return gain_corrections


def bad_corrections(corr_ut):
    """ Placeholder function for testing if the quality of calculated gain 
    corrections are insufficient """
    #FIXME not implemented
    return False


def parse_vlbi_cat(vlbi_cat_fn, ref_ant=None, content=None):
    """ Parses the provided VLBI catalogue file and provides a list of scan parameters and
    a katpoint catalogue of unique targets in the observation. 

    Parameters
    ----------
    vlbi_cat_fn : string
        Location of text file containing observation scan information
    ref_ant : katpoint Antenna object
        Reference antenna for katpoint Catalogue

    Returns
    -------
    obs_params : Dictionary
        Dictionary with observation parameters.
    scan_params : list of dictionaries
        List of dictionaries with parameters for each scan.
    vlbi_cat : katpoint Catalogue object
        katpoint catalogue with unique targets to be observed.

    """
    # Read the VLBI catalogue file 
    with (io.StringIO(content.decode('utf-8')) if content is not None
          else open(vlbi_cat_fn, 'r')) as cat_file:
        csv_lines = [line for line in cat_file.readlines()]
        # Read header info
        obs_params = {}
        header_keys = ['EXPERIMENT', 'POL', 'CAL_PREFIX', 'PCC_PREFIX', 'PCC_VERIFICATION']
        header_lines = [line[1:].strip() for line in csv_lines if line.startswith('#')]
        for line in header_lines:
            obs_key, obs_par = line.split(' ')[0], line.split(' ')[1:]
            if obs_key in header_keys:
                obs_params[obs_key] = obs_par[0]
            elif line.startswith('CH'):
                obs_params[obs_key] = obs_par
        # Create list of scan parameters
        scan_params = []
        # Read every valid line in the provided file
        scan_lines = [line for line in csv_lines if not line.startswith('#')]
        for scan_line in scan_lines:
            # Extract katpoint target fields
            kp_target_ln = ', '.join([kp.strip(' ') for kp in scan_line.split(',')[:-3]]) 
            # Store scan parameters
            scan_params.append({'kp_tgt':katpoint.Target(kp_target_ln, antenna=ref_ant),
                                'name':scan_line.split(',')[-3].strip(),
                                'start_iso':scan_line.split(',')[-2],
                                'start_ut':katpoint.Timestamp(scan_line.split(',')[-2]).secs,
                                'duration':int(scan_line.split(',')[-1])}) 
        # Sort scans by start time.
        sort_fcn = lambda e : e['start_ut']
        scan_params.sort(key=sort_fcn)
        # Create a katpoint catalogue for unique targets
        kp_target_list = []
        for sp in scan_params:
            if sp['kp_tgt'] not in kp_target_list:
                kp_target_list.append(sp['kp_tgt'])
        vlbi_cat = katpoint.Catalogue(kp_target_list, antenna=ref_ant)
    return obs_params, scan_params, vlbi_cat


def _get_capture_telstate(session):
    """Require an authoritative capture identity and disable namespace fallback."""
    telstate = getattr(session, 'telstate', None)
    if telstate is None:
        telstate = getattr(session, '_telstate', None)
    if telstate is None:
        raise RuntimeError("Observation session does not expose a telstate handle")
    capture_block_id = getattr(session, 'capture_block_id', None)
    if capture_block_id is None:
        capture_block_id = telstate['sdp_capture_block_id']
    if (not isinstance(capture_block_id, str) or not capture_block_id
            or any(c not in '0123456789' for c in capture_block_id)):
        raise ValueError("A decimal capture_block_id is required for CSV publication")
    return telstate.view(capture_block_id, exclusive=True)


def persist_vlbi_catalogue(session, catalogue_path, experiment, content=None):
    """Publish the executed CSV under the capture's canonical vlbi_catalogue_* keys."""
    catalogue_file = Path(catalogue_path).expanduser().resolve()
    if content is None:
        with catalogue_file.open('rb') as stream:
            content = stream.read(16 * 1024 * 1024 + 1)
    catalogue_content = content
    if not isinstance(catalogue_content, bytes) or not 0 < len(catalogue_content) <= 16 * 1024 * 1024:
        raise ValueError("CSV catalogue must be non-empty and at most 16 MiB")
    catalogue_content.decode('utf-8')
    catalogue_sha256 = hashlib.sha256(catalogue_content).hexdigest()
    telstate = _get_capture_telstate(session)
    key_values = {
        'vlbi_catalogue_content': catalogue_content,
        'vlbi_catalogue_sha256': catalogue_sha256,
        'vlbi_catalogue_name': catalogue_file.name,
        'vlbi_experiment': experiment,
        'vlbi_catalogue_format': 'csv',
    }
    for key, value in key_values.items():
        telstate.add(key, value, immutable=True)


class vlbi_observation_manager():
    """ Class for performing observations according to an absolute time schedule.
    """
    def __init__(self, scan_pars, horizon, min_scan_duration, max_wait_time,
                 max_mpcc_delay):
        """ Initialises schedule manager variables.

        Parameters
        ----------
        scan_pars : list
            List of Dictionary objects containing scan parameters from parsed catalogue
            file. 
        horizon : float
            Value in degrees below which targets will be deemed too low in elevation to
            observe.
        min_scan_duration : float
            Minimum allowed scan duration, in seconds. A scan will be skipped if the 
            estimated scan duration will be shorter than this value.
        max_wait_time : float
            Maximum time telescope will wait for an upcoming scan, in seconds. A scan 
            will be skipped if a scan is further away in future than this value.
        max_mpcc_delay : float
            The maximum time delay beyond the intended start time of a phase correction
            operation that will be accommodated. If the phase correction operation is not
            started within the specified time since its schedule start time, it will be 
            skipped.
        """
        # Set provided parameters
        self.scan_pars = scan_pars
        self.horizon = horizon
        self.n_scans = len(scan_pars)
        self.max_wait_time = max_wait_time
        self.min_scan_duration = min_scan_duration
        self.max_mpcc_delay = max_mpcc_delay
        # Define timing model constants
        self.timing_mdl = {'max_slew':120, 'pipeline':90, 'set_gains':480}
        # Initialise variables for progress reporting
        self.scans_observed = np.zeros(len(scan_pars), dtype=bool)
        self.scans_skipped = np.zeros(len(scan_pars), dtype=bool)
        self.obs_start = np.zeros(len(scan_pars))
        self.obs_stop = np.zeros(len(scan_pars))
        # Initialise current scan info
        self.scan_ind = 0
        self.scan_start = 0 
        self.scan_stop = 0   
        # Parameter indicating whether telescope should slew to a waiting position before scan start.
        self.wait_for_start = False 
        # Set status to observing
        self.observing = True

    def mpcc_feasible(self, current_ut):
        """ Check if the phase correction operation is feasible under constraints.
        
        Parameters
        ----------
        current_ut : float
            Time in UTC seconds since Unix epoch. 
            
        Returns
        -------
        Boolean
            True if the phase correction operation is feasible. 
        """
        # Set scan info
        scan_name = self.scan_pars[self.scan_ind]['name']
        start_time = self.scan_pars[self.scan_ind]['start_ut']
        # Check various conditions to ensure the target is available.
        if (start_time - current_ut > self.max_wait_time):
            # Check if time before scan starts is below max waiting time.
            # - Not assumming you will pre-empt the scan by starting the slew early, since
            #   the target may be below horizon.
            user_logger.info("Skipping {}. Time until scan start exceeds max waiting time.".format(scan_name))
            self._skip_scan()
            return False
        elif (current_ut - start_time > self.max_mpcc_delay):
            # Check if maximum allowed mpcc delay has passed.
            user_logger.info("Skipping {}. Start time has elapsed.".format(scan_name))
            self._skip_scan()
            return False
        else:
            # Set up current target data if there aren't issues
            self.scan_start = start_time
            self.scan_stop = 0
            self.wait_for_start = False
            return True

    def scan_feasible(self, current_ut):
        """ Check if the target at the current scan index is available for observation,
        and if observation is feasible under constraints.
        
        Parameters
        ----------
        current_ut : float
            Time in UTC seconds since Unix epoch. 
            
        Returns
        -------
        Boolean
            True if the current target is available for observation.
        """
        # Set scan info
        scan_name = self.scan_pars[self.scan_ind]['name']
        # Find scan start / stop time adjusted for horizon 
        start_time, stop_time, wait, below_horizon = self._time_above_horizon()
        # Compute estimated slew time.
        slew_time = self._slew_time(current_ut) 
        # Check various conditions to ensure the target is available.
        if below_horizon:
            # Skip scan if target is below horizon.
            user_logger.info("Skipping {}. Target below horizon for entire duration of scan.".format(scan_name))
            self._skip_scan()
            return False
        elif (stop_time - start_time < self.min_scan_duration):
            # Skip scan if available scan time is below limit
            user_logger.info("Skipping {}. Scan duration below minimum.".format(scan_name))
            self._skip_scan()
            return False
        elif (current_ut + slew_time > stop_time - self.min_scan_duration):
            # Skip scan if insufficient time available to observe it
            user_logger.info("Skipping {}. Stop time elapsed or insufficient time remaining.".format(scan_name))
            self._skip_scan()
            return False
        elif (start_time - current_ut > self.max_wait_time):
            # Check if time before scan starts is below max waiting time.
            # - Not assumming you will pre-empt the scan by starting the slew early, since
            #   the target may be below horizon.
            user_logger.info("Skipping {}. Time until scan start exceeds max waiting time.".format(scan_name))
            self._skip_scan()
            return False
        else:
            # Set up current target data if there aren't issues
            self.scan_start = start_time
            self.scan_stop = stop_time
            self.wait_for_start = wait
            return True

    def _time_above_horizon(self, time_buffer=120):
        """ Checks the time the current target will be above horizon.
        Evaluates the entire scan duration on a second interval, returning
        the start time, end time and total number of seconds above horizon.

        Parameters
        ----------
        time_buffer : int
            Buffer time in seconds to add when cropping scan time to accommodate 
            horizon. Done in order to avoid inadvertantly attempting to observe 
            below horizon on account of script delays or variation in antenna 
            positions. 
        
        Returns
        -------
        start_ut : float
            Scan start time adjusted for elevation above horizon, in UTC seconds 
            since Unix epoch.
        stop_ut : float
            Scan stop time adjusted for elevation above horizon, in UTC seconds 
            since Unix epoch.
        below_horizon : boolean
            True if target is below horizon for entire duration of scan.
        """
        # Compute scan time axis at second interval
        scan_start = round(self.scan_pars[self.scan_ind]['start_ut'])
        scan_stop = round(scan_start + self.scan_pars[self.scan_ind]['duration'])
        time_buffer = int(time_buffer)
        t_axis = np.arange(scan_start-time_buffer, scan_stop+time_buffer+1, 1)
        # Compute target elevation
        kp_tgt = self.scan_pars[self.scan_ind]['kp_tgt']
        tgt_el = kp_tgt.azel(timestamp=t_axis)[1]*180/np.pi
        t_axis_valid = t_axis[tgt_el > self.horizon]
        if len(t_axis_valid)==len(t_axis):
            # Target is above horizon for entire scan
            start_ut, stop_ut = scan_start, scan_stop
            wait, below_horizon = False, False
        elif len(t_axis_valid) > 1:
            # Target is above horizon for nonzero duration
            start_ut = t_axis_valid[0+time_buffer]  
            stop_ut = t_axis_valid[-1-time_buffer]
            wait = t_axis_valid[0] != t_axis[0]  
            below_horizon = False
            scan_name = self.scan_pars[self.scan_ind]['name']
            user_logger.info("Adjusted {} scan time to accomodate horizon.".format(scan_name))
            user_logger.info("{} new start time: {}, old: {}".format(scan_name, start_ut, scan_start))
            user_logger.info("{} new stop time: {}, old: {}".format(scan_name, stop_ut, scan_stop))
        else:
            # Target is below horizon for entire scan 
            start_ut, stop_ut = 0, 0
            wait, below_horizon = True, True
        return start_ut, stop_ut, wait, below_horizon

    def _slew_time(self, time_ut):
        """Get estimated slew time to next target. Slew time estimator function modified from astrokat
        simulate at https://github.com/ska-sa/astrokat/blob/master/astrokat/simulate.py
    
        Parameters
        ----------
        time: float
            Unix timestamp at which slew will occur.
    
        Returns
        -------
        slew_time: float
            The number of seconds it takes to slew.
    
        """ 
        # MeerKAT receptor parameters for azimuth and elevation slewing
        # (some from specifications, some from empirical data - see JIRA MT-1206).
        _SLEW_INIT_OVERHEAD = 2.3
        _EL_SPEED_DEG_PER_SEC = 1.0
        _EL_ACCEL_DEG_PER_SEC_SQ = 0.5
        _EL_LONG_SLEW_DEG = 7.0
        _EL_LONG_SLEW_SETTLE_TIME_SEC = 8.1
        _AZ_SPEED_DEG_PER_SEC = 2.0
        _AZ_ACCEL_DEG_PER_SEC_SQ = 1.0
        _AZ_LONG_SLEW_DEG = 0.0
        _AZ_LONG_SLEW_SETTLE_TIME_SEC = 6.1 
        # Get azimuth and elevation of current and next target 
        if self.scan_ind==0:
            # If current pointing direction is unknown, return max slew time.
            return self.timing_mdl['max_slew']
        current_tgt = self.scan_pars[self.scan_ind-1]['kp_tgt']
        next_tgt = self.scan_pars[self.scan_ind]['kp_tgt']
        current_az_rad, current_el_rad = current_tgt.azel(time_ut)
        next_az_rad, next_el_rad = next_tgt.azel(time_ut)
        # Compute az/el distance
        az_dist = np.abs(next_az_rad - current_az_rad)*180/np.pi
        el_dist = np.abs(next_el_rad - current_el_rad)*180/np.pi
        # wrap angle into +-180, ignoring receptor cable wrapping
        if az_dist > 180.0:
            az_dist = np.abs((az_dist + 180.) % 360. - 180.)
        # Time, t, to accelerate to full speed: v = u + at
        t_el = (_EL_SPEED_DEG_PER_SEC - 0.0) / _EL_ACCEL_DEG_PER_SEC_SQ
        t_az = (_AZ_SPEED_DEG_PER_SEC - 0.0) / _AZ_ACCEL_DEG_PER_SEC_SQ
        # Corresponding displacement to accelerate
        # up to full speed:  s = ut + (at^2)/2
        s_el = 0.0 * t_el + (_EL_ACCEL_DEG_PER_SEC_SQ * t_el ** 2) / 2.0
        s_az = 0.0 * t_az + (_AZ_ACCEL_DEG_PER_SEC_SQ * t_az ** 2) / 2.0
        # The factors of 2 account for acceleration and deceleration
        # i.e., ramping up to full speed, and then ramping down to stop
        if el_dist > 2.0 * s_el:
            el_left = el_dist - 2.0 * s_el
            el_slew_time = 2.0 * t_el + el_left / _EL_SPEED_DEG_PER_SEC
            if el_left > _EL_LONG_SLEW_DEG:
                el_slew_time = el_slew_time + _EL_LONG_SLEW_SETTLE_TIME_SEC
        else:
            # Time taken to cover distance: s = ut + (at^2)/2
            s_el = el_dist / 2.0
            el_slew_time = 2.0 * 2.0 * np.sqrt(s_el / _EL_ACCEL_DEG_PER_SEC_SQ)
        # The factors of 2 account for acceleration and deceleration
        if az_dist > 2.0 * s_az:
            az_left = az_dist - 2.0 * s_az
            az_slew_time = 2.0 * t_az + az_left / _AZ_SPEED_DEG_PER_SEC
            if az_left > _AZ_LONG_SLEW_DEG:
                az_slew_time = az_slew_time + _AZ_LONG_SLEW_SETTLE_TIME_SEC
        else:
            # Time taken to cover distance: s = ut + (at^2)/2
            s_az = az_dist / 2.0
            az_slew_time = 2.0 * 2.0 * np.sqrt(s_az / _AZ_ACCEL_DEG_PER_SEC_SQ)
        # Add additional overhead between initialising and slewing
        az_slew_time += _SLEW_INIT_OVERHEAD
        el_slew_time += _SLEW_INIT_OVERHEAD
        return max(az_slew_time, el_slew_time)

    def _skip_scan(self):
        """ Skip current scan and update observation parameters.
        """
        self.scans_skipped[self.scan_ind] = True
        self.scan_start = 0
        self.scan_stop = 0
        self.wait_for_start = False
        if self.scan_ind < self.n_scans-1:
            self.scan_ind += 1
        else:
            self.observing = False

    def get_scan_info(self):
        """ Returns selected parameters from the current scan.
        
        Returns
        -------
        scan_name : string
            Name of the current scan
        scan_duration : float
            Duration of the scan as per the schedule.
        tgt_name : string
            Name of katpoint Target object for the current target.
        """
        tgt_name = self.scan_pars[self.scan_ind]['kp_tgt'].name
        scan_name = self.scan_pars[self.scan_ind]['name']
        scan_duration = self.scan_pars[self.scan_ind]['duration']
        return scan_name, scan_duration, tgt_name

    def get_wait_target(self):
        """ Returns an az/el katpoint target at the az/el coorindates that 
        the current target will have at the start of the scan. Used as a means
        to slew antennas to a waiting position prior to scan start.
        
        Returns
        -------
        wait_tgt : katpoint Target object
            katpoint Target object for the waiting target.
        """
        scan_name = self.scan_pars[self.scan_ind]['name']
        kp_tgt = self.scan_pars[self.scan_ind]['kp_tgt']
        az, el = kp_tgt.azel(timestamp=self.scan_start)
        # Code to avoid cable unwrap slew (not sure if still relevant)
        if (az*180/np.pi > 275.0):
            az -= 2*np.pi
        kp_str = 'Waiting position for %s, azel, %10.8f, %10.8f' % (scan_name, az*180/np.pi, el*180/np.pi)
        wait_tgt = katpoint.Target(kp_str, antenna=kp_tgt.antenna)   
        return wait_tgt

    def time_to_start(self, current_ut):
        """ Return remaining time until current scan start.
        
        Parameters
        ----------
        current_ut : float
            Time in UTC seconds since Unix epoch. 
            
        Returns
        -------
        track_time : float
            Time remaining until current scan start, in seconds.
        """
        dt = self.scan_start - current_ut
        return dt if dt > 0.0 else 0.0
    
    def time_to_stop(self, current_ut):
        """ Return remaining time until current scan stop.
        
        Parameters
        ----------
        current_ut : float
            Time in UTC seconds since Unix epoch. 
            
        Returns
        -------
        track_time : float
            Time remaining until current scan stop, in seconds.
        """
        dt = self.scan_stop - current_ut
        return dt if dt > 0.0 else 0.0

    def scan_done(self, start_ut, stop_ut):
        """ Log scan performed.
        
        Parameters
        ----------
        start_ut : float
            Scan start time in UTC seconds since Unix epoch. 
        stop_ut : float
            Scan stop time in UTC seconds since Unix epoch. 
        """
        self.scans_observed[self.scan_ind] = True 
        self.obs_start[self.scan_ind] = start_ut
        self.obs_stop[self.scan_ind] = stop_ut
        self.scan_start = 0
        self.scan_stop = 0
        self.wait_for_start = False
        if self.scan_ind < self.n_scans-1:
            self.scan_ind += 1
        else:
            self.observing = False        

    def obs_report(self):
        """ Outputs a summary of observed targets.
        """
        if np.sum(self.scans_observed)>0:
            user_logger.info("Scans Observed:")
            for ii in range(self.n_scans):
                if self.scans_observed[ii]:
                    scan_name = self.scan_pars[ii]['name']
                    tgt_name = self.scan_pars[ii]['kp_tgt'].name
                    duration = self.obs_stop[ii] - self.obs_start[ii]
                    user_logger.info("- {}: {} for {:.1f} sec".format(scan_name, tgt_name, duration))
        else:
            user_logger.info("No scans observed.")
            

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    
    # Set up beamformer options prior to session start
    user_logger.info('Performing beamformer setup.')
    # Get list of antennas to enable for beamformer
    cbf = SessionCBF(kat)
    bf_ants = [ant.name for ant in kat.ants]
    # Keep only core antennas
    if opts.bf_core_only:
        bf_ants = [ant_name for ant_name in bf_ants if ant_name in core_ants]
    user_logger.info('Selected %d of %d antennas',
                     len(bf_ants), len(kat.ants))
    user_logger.info('Antennas in beamformer = %s', ' '.join(sorted(bf_ants)))
    # Set the beamformer quantisation gain and beamformer weights
    for stream in cbf.beamformers:
        stream_inst = str(stream).split(' ')[1].split('.')[0][1:]
        if stream_inst==opts.bf_inst:
            # Set beamformer quantization gain.
            bq_gain = opts.bf_gain / np.sqrt(len(bf_ants))
            reply = stream.req.quant_gains(bq_gain)
            if reply.succeeded:
                user_logger.info('Beamformer %s quantization gain set to %g for %d antennas.',
                                 stream, bq_gain, len(bf_ants))
            else:
                user_logger.warning('Failed to set beamformer quantization gain %g on '
                                    'stream %s!', bq_gain, stream)
            # Set the beamformer weights.
            user_logger.info('Setting beamformer weights for stream %s:', stream)
            weights = []
            for inp in stream.inputs:
                weight = 1.0 if inp[:-1] in bf_ants else 0.0
                weights.append(weight)
                user_logger.info('  input %r will get weight %f', inp, weight)
            reply = stream.req.weights(*weights)
            if reply.succeeded:
                user_logger.info('Set input weights successfully.')
            else:
                user_logger.warning('Failed to set input weights!')
    
    # Set reference antenna 
    ref_ant = kat.sources.antenna
    # Read observation scan parameters and catalogue
    # Freeze the input used for both execution and downstream publication.
    with Path(args[0]).open('rb') as catalogue_file:
        catalogue_content = catalogue_file.read(16 * 1024 * 1024 + 1)
    if not 0 < len(catalogue_content) <= 16 * 1024 * 1024:
        raise ValueError('CSV catalogue must be non-empty and at most 16 MiB')
    exp_pars, scan_pars, obs_cat = parse_vlbi_cat(args[0], ref_ant, content=catalogue_content)
    mcal_scan_name = exp_pars['CAL_PREFIX'].lower()
    mpcc_scan_name = exp_pars['PCC_PREFIX'].lower()
    mpcc_verification_scans = True if exp_pars['PCC_VERIFICATION']=='True' else False
    
    # Initialise observation schedule manager
    obs_man = vlbi_observation_manager(scan_pars, opts.horizon, opts.min_scan_duration, 
                                       opts.max_wait_time, opts.max_mpcc_delay)
    
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
            
        # Initialise session
        session.standard_setup(**vars(opts))               
        # Start SDP capture.  
        session.capture_start()
        # Persist the exact per-observation catalogue used by this script into
        # capture-scoped vlbi_catalogue_* keys for downstream processing.
        persist_vlbi_catalogue(session, args[0], exp_pars['EXPERIMENT'], content=catalogue_content)
            
        #FIXME Loop counters to prevent infinite loop errors.
        loop_count, max_loops = 0, 10*len(scan_pars) 
        
        # Target observation loop 
        while obs_man.observing:
            
            #FIXME Loop counters to prevent infinite loop errors.
            loop_count += 1 
            
            # Read current scan info
            scan_name, scan_duration, tgt_name = obs_man.get_scan_info()
            obs_tgt = obs_cat[tgt_name]
            
            # Check if phase-up corection is required
            if scan_name.startswith(mpcc_scan_name):
    
                # Perform phase correction if feasible
                if obs_man.mpcc_feasible(time.time()):
                    
                    user_logger.info("Peforming phase correction {}.".format(scan_name))
                    # Create targets with appropriate tags for calibration and tracking
                    calib_tgt = katpoint.Target(obs_tgt.format_katcp())
                    calib_tgt.add_tags('radec bfcal single_accumulation')
                    dwell_tgt = katpoint.Target(obs_tgt.format_katcp())
                    dwell_tgt.tags = ['radec']
                    # Resetting wideband gains.
                    if opts.reset_gains:
                        fengine_gain = session.set_fengine_gains('default')
                    # Perform calibrator track
                    user_logger.info("Initiating %g-second track on target %r",
                                     scan_duration, tgt_name)
                    session.label(scan_name+' track')
                    session.track(calib_tgt, duration=0, announce=True)
                    scan_start = time.time()
                    session.fire_noise_diode(on=scan_duration, off=0)
                    # Wait for pipeline solutions
                    session.label(scan_name+' solve')
                    session.track(dwell_tgt, duration=0, announce=False)
                    user_logger.info("Waiting for gains to materialise in cal pipeline.")
                    # Determine corrections
                    hv_gains, hv_delays = {}, {}
                    timeout = 60 + scan_duration
                    # Wait for the most recent calibration solutions
                    try:
                        hv_gains = session.get_cal_solutions('BCROSS_DIODE_SKY', 
                                                             timeout=timeout, start_time=scan_start)
                        hv_delays = session.get_cal_solutions('KCROSS_DIODE', start_time=scan_start)
                        gains = session.get_cal_solutions('G', start_time=scan_start)
                        bp_gains = session.get_cal_solutions('B', start_time=scan_start)
                        delays = session.get_cal_solutions('K', start_time=scan_start)
                        cal_channel_freqs = session.get_cal_channel_freqs()
                        # Add HV delay to the usual delay
                        for inp in sorted(delays):
                            delays[inp] += hv_delays.get(inp, 0.0)
                            if np.isnan(delays[inp]):
                                user_logger.warning("Delay fit failed on input %s (all its "
                                                    "data probably flagged)", inp)
                        # Add HV phase to bandpass phase
                        for inp in bp_gains:
                            bp_gains[inp] *= hv_gains.get(inp, 1.0)
                        bp_gains = clean_bandpass(bp_gains, cal_channel_freqs, max_gap_Hz=opts.max_gap_MHz*1e6)
                    except CalSolutionsUnavailable as err:
                        user_logger.warning("Failed to obtain calibration solutions "
                                            "from pipeline: %s", err)
                        user_logger.info("Proceeding without performing gain corrections.")
                        # Mark scan observed 
                        obs_man.scan_done(scan_start, time.time())
                        # Skip the rest of the correction operations. 
                        continue
                    # Calculate and apply corrections
                    if not kat.dry_run:
                        # Calculate wideband corrections
                        corrections = calculate_corrections(gains, bp_gains, delays,
                                                            cal_channel_freqs, fengine_gain)
                        if bad_corrections(corrections):
                            user_logger.warning("Calculated gain corrections failed "
                                                "a quality check.")
                            user_logger.info("Proceeding without performing gain corrections.")
                            # Mark scan observed 
                            obs_man.scan_done(scan_start, time.time())
                            # Skip the rest of the correction operations.
                            continue
                        session.label(scan_name+' apply')
                        session.track(dwell_tgt, duration=0, announce=False)
                        user_logger.info("Setting F-engine gains to phase up antennas.")
                        # Apply wideband corrections
                        if not opts.skip_wide_correction:
                            session.set_fengine_gains(corrections)
                        # Interpolate wideband corrections onto narrowband frequencies 
                        nb_center_freq = np.round(opts.nb_center_freq/1712e6*2**32)*1712e6/2**32
                        freqs_nb = nb_center_freq + (np.arange(2**15)-2**14)*(107e6/2**15)
                        if not (cal_channel_freqs[0]<=freqs_nb[0] and freqs_nb[-1]<=cal_channel_freqs[-1]):
                            user_logger.warning("Specified narrowband frequency axis not fully contained within sampled band.")
                        corrections_nb = {}
                        for inp in corrections:
                            if type(corrections[inp]) is np.ndarray:
                                # Interpolate corrections
                                corrections_nb[inp] = np.interp(freqs_nb, cal_channel_freqs, corrections[inp])
                                # Adjust corrections to conform to narrowband default gains.
                                corrections_nb[inp] *= rqgain_narrow/rqgain_wide
                            elif corrections[inp]==0.0:
                                # Propagate zeroed antennas to narrowband
                                corrections_nb[inp] = 0.0
                        # Apply narrowband corrections
                        session.set_fengine_gains(corrections_nb, instrument_name='narrow1')
                    if mpcc_verification_scans:   
                        # Run verification scan
                        user_logger.info("Revisiting target {} for {} seconds to verify phase-up.".format(tgt_name, scan_duration))
                        session.label(scan_name+' verify')
                        session.track(calib_tgt, duration=0, announce=False)
                        session.fire_noise_diode(on=scan_duration, off=0)
                    # Mark scan observed and log observation parameters
                    obs_man.scan_done(scan_start, time.time())
                
            # Check if current catalogue entry can be observed
            elif obs_man.scan_feasible(time.time()):
                            
                # Check whether target is above horizon yet
                wait_for_start = (obs_man.time_to_start(time.time()) > 0) and obs_man.wait_for_start
                if wait_for_start:
                	# Set compscan label to waiting
                	session.label("Wait for {}".format(scan_name))
                	# Slew to anticipated az/el position at start time.
                	wait_tgt = obs_man.get_wait_target()
                	session.track(wait_tgt, duration=0, announce=True)
                	# Wait until target is above horizon
                	wait_time_sec = obs_man.time_to_start(time.time())
                	if wait_time_sec > 0:
                		session.track(wait_tgt, duration=wait_time_sec, announce=True)
                # Slew to target and start tracking
                session.label(scan_name)
                session.track(obs_tgt, duration=0, announce=True)
                # Track time until necessary UTC stop time
                obs_duration = obs_man.time_to_stop(time.time())
                scan_start = time.time()
                user_logger.info("Observing scan: {}".format(scan_name))
                if scan_name.startswith(mcal_scan_name):
                	# Fire noise diode during calibration scans
                	session.fire_noise_diode(on=obs_duration, off=0)
                else:
                	session.track(obs_tgt, duration=obs_duration)
                # Mark target observed and log observation parameters
                obs_man.scan_done(scan_start, time.time())
    
            #FIXME A condition to break out of an infinite loop error.
            if loop_count> max_loops:
                user_logger.info("ERROR: Breaking out of infinite loop...")
                break
    
        obs_man.obs_report()
