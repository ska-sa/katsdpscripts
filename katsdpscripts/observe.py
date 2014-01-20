###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
"""Set of useful routines to do standard observations with KAT."""

import optparse
import uuid

import katpoint

from .defaults import colors, user_logger
from .utility import tbuild
from .conf import get_system_configuration, configure_core
from .kat7_session import CaptureSession as KAT7CaptureSession
from .kat7_session import TimeSession as KAT7TimeSession
from .rts_session import CaptureSession as RTSCaptureSession
from .rts_session import TimeSession as RTSTimeSession
from .rts_session import projections, default_proj, ant_array


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

    parser.add_option('--sb-id-code', type='string',
                      help='Schedule block id code for observation, '
                           'required in order to allocate correct resources')
    parser.add_option('-u', '--experiment-id',
                      help='Experiment ID used to link various parts of '
                      'experiment together (use sb-id-code by default, or random UUID)')
    parser.add_option('-o', '--observer',
                      help='Name of person doing the observation (**required**)')
    parser.add_option('-d', '--description', default='No description.',
                      help="Description of observation (default='%default')")
    parser.add_option('-f', '--centre-freq', type='float', default=1822.0,
                      help='Centre frequency, in MHz (default=%default)')
    parser.add_option('-r', '--dump-rate', type='float', default=1.0,
                      help='Dump rate, in Hz (default=%default)')
# This option used to be in observe1, but did not make it to the
# common set of options of observe1 / observe2
#    parser.add_option('-w', '--discard-slews', dest='record_slews', action='store_false', default=True,
#                      help='Do not record all the time, i.e. pause while antennas are slewing to the next target')
    parser.add_option('-n', '--nd-params', default='coupler,10,10,180',
                      help="Noise diode parameters as '<diode>,<on>,<off>,<period>', "
                      "in seconds or 'off' for no noise diode firing (default='%default')")
    parser.add_option('-p', '--projection', type='choice',
                      choices=projections, default=default_proj,
                      help="Spherical projection in which to perform scans, "
                      "one of '%s' (default), '%s'" % (
                          projections[0], "', '".join(projections[1:])))
    parser.add_option('-y', '--dry-run', action='store_true', default=False,
                      help="Do not actually observe, but display script "
                      "actions at predicted times (default=%default)")
    parser.add_option('--stow-when-done', action='store_true', default=False,
                      help="Stow the antennas when the capture session ends")
    parser.add_option('--mode',
                      help="DBE mode to use for experiment, keeps current mode by default)")
    parser.add_option('--dbe-centre-freq', type='float', default=None,
                      help="DBE centre frequency in MHz, used to select coarse band for "
                           "narrowband modes (unchanged by default)")
    parser.add_option('--horizon', type='float', default=5.0,
                      help="Session horizon (elevation limit) in degrees (default=%default)")
    parser.add_option('--no-mask', action='store_true', default=False,
                      help="Keep all correlation products by not applying baseline/antenna mask")

    return parser


def verify_and_connect(opts):
    """Verify command-line options, build KAT configuration and connect to devices.

    This inspects the parsed options and requires at least *observer* and
    *system* to be set. It generates an experiment ID if missing (for now using
    the sb-id-code if that is available) and verifies noise diode parameters if
    given. It then creates a KAT connection based on the *system* option,
    reusing an existing connection or falling back to the local system if
    required. The resulting KATCoreConn object is returned.

    Parameters
    ----------
    opts : :class:`optparse.Values` object
        Parsed command-line options (will be updated by this function). Should
        contain at least the options *observer* and in future
        *sb-id-code*.

    Returns
    -------
    kat : :class:`utility.KATCoreConn` object
        KAT connection object associated with this experiment

    Raises
    ------
    ValueError
        If required options are missing

    """
    # Various non-optional options...
    if not hasattr(opts, 'observer') or opts.observer is None:
        raise ValueError('Please specify the observer name via -o option '
                         '(yes, this is a non-optional option...)')
    # if not hasattr(opts, 'sb_id_code') or opts.sb_id_code is None:
    #    raise ValueError('Please specify the --sb-id-code option '
    #                     '(yes, this is a non-optional option...)')

    # For now we force the sb-id-code as the experiment_id as no one is using it
    # This will change in future
    if opts.sb_id_code is not None:
        opts.experiment_id = opts.sb_id_code
    elif not hasattr(opts, 'experiment_id') or opts.experiment_id is None:
        # Generate unique string via RFC 4122 version 1
        opts.experiment_id = str(uuid.uuid1())

    site, system = get_system_configuration()

    # If given, verify noise diode parameters (should be 'string,number,number,number') and convert to dict
    if hasattr(opts, 'nd_params'):
        # Shortcut for switching off noise diodes
        if opts.nd_params.lower() == 'off':
            opts.nd_params = 'coupler,0,0,-1'
        try:
            opts.nd_params = eval("{'diode':'%s', 'on':%s, 'off':%s, 'period':%s}" %
                                  tuple(opts.nd_params.split(',')), {})
        except (TypeError, NameError):
            raise ValueError("Noise diode parameters are incorrect (should be 'diode,on,off,period')")
        for key in ('on', 'off', 'period'):
            if opts.nd_params[key] != float(opts.nd_params[key]):
                raise ValueError("Parameter nd_params['%s'] = %s (should be a number)" % (key, opts.nd_params[key]))

    # Build KAT configuration which connects to all the proxies and devices and queries their commands and sensors
    try:
        if opts.sb_id_code is not None:
            kat = configure_core(sb_id_code=opts.sb_id_code, dry_run=opts.dry_run)
        else:
            # Temporarily give the user override options
            print colors.Red, "\nBuilding without a schedule block id code is deprecated." \
                  "\nTHERE MAY BE CONTROL CLASHES!!!!\nBut for one last time we will allow it ...", colors.Normal
            choice = raw_input(colors.Red + "Do you want to cancel this build? y/n ...." + colors.Normal)
            if choice not in ['n', 'N']:
                raise ValueError("Cancelled build of KATCoreConn object connection for site=%s system=%s" % (site, system))
            kat = tbuild(system=system, conn_clients='all', controlled_clients='all')
        user_logger.info("Using KAT connection with configuration=%s "
                         "sb_id_code=%s\nControlled objects: %s" %
                         (kat.system, opts.sb_id_code, kat.controlled_objects))
    except ValueError, err:
        # Don't default to local build.
        kat = None
        user_logger.error("Could not build KATCoreConn object connection for site=%s system=%s (%s)" % (site, system, err))
        raise ValueError("Could not build KATCoreConn object connection for site=%s system=%s (%s)" % (site, system, err))

    return kat


def start_session(kat, **kwargs):
    """Start capture session (real or fake).

    This starts a capture session initialised with the given arguments, choosing
    the appropriate session class to use based on the arguments. The system is
    inspected to determine which version of :class:`CaptureSession` to use,
    while the kat.dry_run flag decides whether a fake :class:`TimeSession` will
    be used instead.

    Parameters
    ----------
    kat : :class:`utility.KATCoreConn` object
        KAT connection object associated with this experiment
    kwargs : dict, optional
        Ignore any other keyword arguments (simplifies passing options as dict)

    Returns
    -------
    session : :class:`CaptureSession` or :class:`TimeSession` object
        Session object associated with started session

    """
    if hasattr(kat, 'dbe7'):
        return KAT7TimeSession(kat, **kwargs) if kat.dry_run else KAT7CaptureSession(kat, **kwargs)
    else:
        return RTSTimeSession(kat, **kwargs) if kat.dry_run else RTSCaptureSession(kat, **kwargs)


def collect_targets(kat, args):
    """Collect targets specified by name, description string or catalogue file.

    Parameters
    ----------
    kat : :class:`utility.KATCoreConn` object
        KAT connection object associated with this experiment
    args : list of strings
        Argument list containing mixture of target names, description strings
        and / or catalogue file names

    Returns
    -------
    targets : :class:`katpoint.Catalogue` object
        Catalogue containing all targets found

    Raises
    ------
    ValueError
        If final catalogue is empty

    """
    from_names = from_strings = from_catalogues = num_catalogues = 0
    targets = katpoint.Catalogue(antenna=kat.sources.antenna)
    for arg in args:
        try:
            # First assume the string is a catalogue file name
            count_before_add = len(targets)
            try:
                targets.add(file(arg))
            except ValueError:
                user_logger.warning("Catalogue %r contains bad targets" % (arg,))
            from_catalogues += len(targets) - count_before_add
            num_catalogues += 1
        except IOError:
            # If the file failed to load, assume it is a name or description string
            # With no comma in target string, assume it's the name of a target to be looked up in standard catalogue
            if arg.find(',') < 0:
                target = kat.sources[arg]
                if target is None:
                    user_logger.warning("Unknown target or catalogue %r, skipping it" % (arg,))
                else:
                    targets.add(target)
                    from_names += 1
            else:
                # Assume the argument is a target description string
                try:
                    targets.add(arg)
                    from_strings += 1
                except ValueError, err:
                    user_logger.warning("Invalid target %r, skipping it [%s]" % (arg, err))
    if len(targets) == 0:
        raise ValueError("No known targets found in argument list")
    user_logger.info("Found %d target(s): %d from %d catalogue(s), %d from default catalogue and %d as target string(s)"
                     % (len(targets), from_catalogues, num_catalogues, from_names, from_strings))
    return targets
