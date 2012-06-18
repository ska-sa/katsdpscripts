#! /usr/bin/python
#
# Example script illustrating the interactive use of a CaptureSession.
#
# Ludwig Schwardt
# 20 May 2011
#

# The KAT-7 / Fringe Finder system is addressed via this package.
import katuilib

# We now have to connect to the appropriate system (e.g. the real Karoo system,
# a lab test system or even a local one on our own machine). From the ipython
# shell we usually run configure() to automatically select the correct system
# based on the machine we are on.
configure(sb_id_code="20120101-0101")

# Alternatively, we can build a connection to a specific system if we know its
# configuration file.
# kat = katcorelib.tbuild('systems/local.conf')

# The end result is that we have a 'kat' object used to talk to the system.

# Now start a capture session, which creates an HDF5 output file.
# You can optionally set the DBE proxy to use, in order to select the appropriate
# correlator. Currently we have 'dbe' for the Fringe Finder correlator and 'dbe7'
# for the KAT-7 correlator. The function returns a CaptureSession object, which
# is used to manage the session.
session = katuilib.start_session(kat, dbe='dbe7')

# Now you can perform basic setup of the system for the experiment, which is
# simplified by the standard_setup command. At the minimum you need to specify
# which antennas you want to use, your name and a short description of the
# experiment. The antennas can be specified in many ways:
# - a comma-separated list of antenna names: 'ant1,ant2'
# - an antenna device: kat.ant3
# - a list of antenna devices: [kat.ant1, kat.ant2]
# - an antenna device array: kat.ants
# - the keyword 'all' to use all antennas in the system
# You can optionally also set the dump rate in Hz, the centre frequency in MHz,
# the noise diode firing strategy (which diode to use and how often to fire it
# on and off during the canned commands, in seconds) and the elevation limit
# (session horizon) in degrees. If you don't specify these settings, they are
# left unchanged (except for the dump rate, which will be set).
session.standard_setup(ants='ant1,ant2', observer='me', description='Testing testing...',
                       centre_freq=1822.0, dump_rate=1.0,
                       nd_params={'diode' : 'coupler', 'on' : 10.0, 'off' : 10.0, 'period' : 180.},
                       horizon=5.0)

# The data actually starts flowing once you call capture_start(). [Note: on the
# Fringe Finder system this action is postponed even further, until you actually
# call a standard session action such as track, scan or fire_noise_diode, as the
# recording can only start once the details of the first compound scan is known.]
session.capture_start()

# Now you want to do some real observing. Drive the dishes around, fire noise
# diodes, etc. You can send any katcp request as usual at this time via the kat object:
target = kat.sources['Ori A']
kat.ant1.req.target(target)
kat.ant1.req.mode('POINT')

# The session object provides some useful canned command sequences that are
# used extensively by the standard scripts. Here are some examples - please find
# out more about them by using the ipython ? operator and tab completion (e.g.
# type 'session.raster_scan?' to see the multitude of parameters of this method).
# Note that these are all blocking calls, meaning that they will only return
# control back to the user prompt once the action is fully completed.

# Checks whether all antennas are locked on the specified target
session.on_target(target)
# Checks whether the target remains above session horizon for the next 30 seconds for all antennas
session.target_visible(target, duration=30.)
# Switch the coupler diode on for 10 seconds
session.fire_noise_diode(diode='coupler', on=10.0)
# Track the target for 10 seconds
session.track(target, duration=10.0)
# Perform a scan across the target in azimuth with a span of 6 degrees, lasting 10 seconds
session.scan(target, duration=10.0, start=(-3.0, 0.0), end=(3.0, 0.0))
# For the grand finale, perform a raster scan on the target, consisting of 3 scans lasting 10 seconds
# each and spanning 6 degrees in azimuth, with a 0.5-degree spacing between scans in elevation
# Also indicate that this is a new compound scan by setting the session label
session.label('raster')
session.raster_scan(target, num_scans=3, scan_duration=10.0, scan_extent=6.0, scan_spacing=0.5)

# Finally, stop the correlator and close off the HDF5 file when you are done.
# In the standard scripts, this step is done implicitly via the Python "with"
# statement, which ensures that the file is properly closed when an error occurs
# or the user gets bored and kills the session with Ctrl-C (i.e. very useful!).
# To do this, run the above observation commands in a block like so:
#
# with katuilib.start_session(kat, ...) as session:
#     session.standard_setup(...)
#     session.capture_start()
#     ... other observation commands ...
#
# The downside of this code is that it is not interactive due to its block nature,
# and that's why we are taking a more interactive approach in this example.
session.end()

# If you want to know the name of your HDF5 file after you are done, you can do this:
print session.output_file
