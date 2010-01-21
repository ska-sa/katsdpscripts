"""Set of useful routines to do observations with Fringe Finder."""

import numpy as np

def setup(ff, ants, centre_freq=1800.0, dump_rate=1.0):
    """Initialise Fringe Finder hardware before the observation starts.

    This prepares the data capturing subsystem (DBE and k7writer) and sets the
    RFE7 LO frequency.

    Parameters
    ----------
    ff : FF object
        Fringe Finder connection object
    ants : List of antenna device objects, or string
        List of antennas that will perform scans, as FFDevice objects,
        e.g. [ff.ant1, ff.ant2]. If this is 'all', use all antennas.
    centre_freq : float
        RF centre frequency, in MHz
    dump_rate : float
        Correlator dump rate, in Hz

    """
    if ants == 'all':
        ants = ff.ants.devs
    # Create list of baselines, based on the selected antennas
    # (1 = ant1-ant1 autocorr, 2 = ant1-ant2 cross-corr, 3 = ant2-ant2 autocorr)
    # This determines which HDF5 files are created
    if ff.ant1 in ants and ff.ant2 in ants:
        baselines = [1, 2, 3]
    elif ff.ant1 in ants and ff.ant2 not in ants:
        baselines = [1]
    elif ff.ant1 not in ants and ff.ant2 in ants:
        baselines = [3]
    else:
        baselines = []

    # Start with a clean state, by stopping the DBE
    ff.dbe.req.capture_stop()

    # Set centre frequency in RFE stage 7
    ff.rfe7.req.rfe7_lo1_frequency(4200.0 + centre_freq, 'MHz')
    effective_lo_freq = (centre_freq - 200.0) * 1e6

    # Set data output directory (typically on ff-dc machine)
    ff.dbe.req.k7w_output_directory("/var/kat/data")
    # Tell k7_writer to write the selected baselines to HDF5 files
    ff.dbe.req.k7w_baseline_mask(*baselines)
    ff.dbe.req.k7w_write_hdf5(1)

    # This is a precaution to prevent bad timestamps from the correlator
    ff.dbe.req.dbe_sync_now()
    # The DBE proxy needs to know the dump rate as well as the effective LO freq, which is used for fringe stopping (eventually)
    ff.dbe.req.capture_setup(1000.0 / dump_rate, effective_lo_freq)

def raster_scan(ff, ants, target, num_scans=3, scan_duration=20.0,
                scan_extent=4.0, scan_spacing=0.5, compscan_id=0,
                scan_in_azimuth=True, drive_strategy="shortest-slew"):
    """Perform raster scan on target.

    This performs a series of scans across a target, by scanning in either
    azimuth or elevation, while the other coordinate is changed in steps for
    each scan. Each scan is offset by the same amount on both sides of the
    target along the scanning coordinate (and therefore has the same extent),
    and the scans are arranged symmetrically around the target in the
    non-scanning (stepping) direction. If an odd number of scans are done, the
    middle scan will therefore pass directly over the target. The default is
    to scan in azimuth and step in elevation, leading to a series of horizontal
    scans. Each scan is scanned in the opposite direction to the previous scan
    to save time. Additionally, the first scan always starts at the top left
    of the target, regardless of scan direction.

    In addition to the proper scans across the source (labelled 'scan' in the
    dataset), data is also recorded while the antennas are moving to the start
    of the next scan. (This is due to the design of the data capturing system,
    which prefers to run continuously.) These segments are labelled 'slew' and
    will typically be discarded during processing.

    All scans in the raster scan are grouped together in a single compound scan
    in the HDF5 data file, as they share the same target. Additionally, the
    compound scan index may be specified. If more than one raster scan is done
    on different targets during the same observation, each raster scan should
    have a different compound scan index, as each compound scan can only have
    a single target associated with it.

    The antennas that will perform the raster scan should be specified, to
    provide some reminder to the user of what will physically move. They all
    perform the same raster scan across the given target, in parallel.

    Parameters
    ----------
    ff : FF object
        Fringe Finder connection object
    ants : List of antenna device objects, or string
        List of antennas that will perform scans, as FFDevice objects,
        e.g. [ff.ant1, ff.ant2]. If this is 'all', use all antennas.
    target : string
        Target to scan across, as description string
    num_scans : integer
        Number of scans across target (an odd number is better, as this will
        scan directly over the source during the middle scan)
    scan_duration : float
        Duration of each scan across target, in seconds
    scan_extent : float
        Extent (angular length) of scan along scanning coordinate, in degrees
        (see *Notes* below)
    scan_spacing : float
        Separation between each consecutive scan along the coordinate that is
        not scanned but stepped, in degrees
    compscan_id : integer
        Compound scan ID number (usually starts at 0)
    ants : List of antenna device objects, optional
        List of antennas that will perform scans, as FFDevice objects,
        e.g. [ff.ant1, ff.ant2] (default = all antennas)
    scan_in_azimuth : {True, False}
        True if azimuth changes during scan while elevation remains fixed;
        False if scanning in elevation and stepping in azimuth instead
    drive_strategy : {'shortest-slew', 'longest-track'}
        Drive strategy employed by antennas, used to decide what to do when
        target is in azimuth overlap region of antenna. The default is to
        go to the wrap that is nearest to the antenna's current position,
        thereby saving time.

    Notes
    -----
    Take note that scanning is done in a projection on the celestial sphere,
    and the scan extent and spacing apply to the projected coordinates.
    The azimuth coordinate of a scan in azimuth will therefore change more
    than the *scan_extent* parameter suggests, especially at high elevations.
    This ensures that the same scan parameters will lead to the same
    qualitative scan for any position on the celestial sphere.

    """
    if ants == 'all':
        ants = ff.ants.devs
    # Initialise antennas
    for ant_x in ants:
        # Set the drive strategy for how antenna moves between targets
        ant_x.req.drive_strategy(drive_strategy)
        # Set the antenna target
        ant_x.req.target(target)

    # Provide target to k7_writer, which will put it in data file (do this *before* changing compound scan ID...)
    ff.dbe.req.k7w_target(target)
    # Set the compound scan ID, which creates a new CompoundScan group in the HDF5 file
    ff.dbe.req.k7w_compound_scan_id(compscan_id)
    # Provide target to the DBE proxy, which will use it as delay-tracking center
    ff.dbe.req.target(target)

    # Create start positions of each scan, based on scan parameters
    scan_steps = np.arange(-(num_scans // 2), num_scans // 2 + 1)
    scanning_coord = (scan_extent / 2.0) * (-1) ** scan_steps
    stepping_coord = scan_spacing * scan_steps
    # These minus signs ensure that the first scan always starts at the top left of target
    scan_starts = zip(scanning_coord, -stepping_coord) if scan_in_azimuth else zip(stepping_coord, -scanning_coord)

    # Iterate through the scans across the target
    for scan_count, scan in enumerate(scan_starts):

        print "Slewing to start of scan %d of %d" % (scan_count + 1, len(scan_starts))
        # Set the new scan ID - this will create a new Scan group in the HDF5 file
        ff.dbe.req.k7w_scan_id(2*scan_count, 'slew')
        # If we haven't yet, start recording data from the correlator
        if ff.dbe.sensor.capturing.get_value() == '0':
            ff.dbe.req.capture_start()
        # Send each antenna to the start position of the next scan
        for ant_x in ants:
            if scan_in_azimuth:
                ant_x.req.scan_asym(scan[0], scan[1], -scan[0], scan[1], scan_duration)
            else:
                ant_x.req.scan_asym(scan[0], scan[1], scan[0], -scan[1], scan_duration)
            ant_x.req.mode("POINT")
        # Wait until they are all in position (with 5 minute timeout)
        for ant_x in ants:
            ant_x.wait("lock", True, 300)

        print "Starting scan %d of %d" % (scan_count + 1, len(scan_starts))
        # Start a new Scan group in the HDF5 file, this time labelled as a proper 'scan'
        ff.dbe.req.k7w_scan_id(2*scan_count + 1, "scan")
        # Start scanning the antennas
        for ant_x in ants:
            ant_x.req.mode("SCAN")
        # Wait until they are all finished scanning (with 5 minute timeout)
        for ant_x in ants:
            ant_x.wait("scan_status", "after", 300)

def shutdown(ff):
    """Stop data capturing to shut down observation.

    Parameters
    ----------
    ff : FF object
        Fringe Finder connection object

    """
    # Obtain the names of the files currently being written to
    files = ff.dbe.req.k7w_get_current_files(tuple=True)[1][2]
    print "Scans complete, data captured to", files

    # Stop the data capture and close the Fringe Finder connections
    ff.dbe.req.capture_stop()
