#!/usr/bin/python
# Script that uses scape to reduce data consisting of scans across multiple point sources.
#

#################################################### Main function ####################################################
import optparse

from katsdpscripts.reduction.analyse_point_source_scans import analyse_point_source_scans

# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                               description="This processes an HDF5 dataset and extracts fitted beam parameters "
                                           "from the compound scans in it. It runs interactively by default, "
                                           "which allows the user to inspect results and discard bad scans.")
parser.add_option("-a", "--baseline", default='sd',
                  help="Baseline to load (e.g. 'ant1' for antenna 1 or 'ant1,ant2' for 1-2 baseline), "
                       "default is first single-dish baseline in file")
parser.add_option("-b", "--batch", action="store_true",
                  help="Flag to do processing in batch mode without user interaction")
parser.add_option("-c", "--channel-mask", default=None, help="Optional pickle file with boolean array specifying channels to mask (default is no mask)")
parser.add_option("-f", "--freq-chans",
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 50% of the bandpass)")
parser.add_option("-k", "--keep", dest="keepfilename",
                  help="Name of optional CSV file used to select compound scans from dataset (implies batch mode)")
parser.add_option("-m", "--monte-carlo", dest="mc_iterations", type='int', default=1,
                  help="Number of Monte Carlo iterations to estimate uncertainty (20-30 suggested, default off)")
parser.add_option("-n", "--nd-models", help="Name of optional directory containing noise diode model files")
parser.add_option("-o", "--output", dest="outfilebase",
                  help="Base name of output files (*.csv for output data and *.log for messages, "
                       "default is '<dataset_name>_point_source_scans')")
parser.add_option("-p", "--pointing-model",
                  help="Name of optional file containing pointing model parameters in degrees")
parser.add_option("-s", "--plot-spectrum", action="store_true", help="Flag to include spectral plot")
parser.add_option("-t", "--time-offset", type='float', default=0.0,
                  help="Time offset to add to DBE timestamps, in seconds (default = %default)")
parser.add_option("-u", "--ku-band", action="store_true", help="Force center frequency to be 12500.5 MHz")
parser.add_option("--old-loader", action="store_true", help="Use old SCAPE loader to open HDF5 file instead of katfile")
parser.add_option("--keep-all", action="store_true", help="Keep all the results if there is a beam that has been fitted")
parser.add_option("--remove-spikes", action="store_true", help="Use the SCAPE method to remove spikes in the passband",default=False)

(opts, args) = parser.parse_args()

if len(args) != 1 or not args[0].endswith('.h5'):
    raise RuntimeError('Please specify a single HDF5 file as argument to the script')

analyse_point_source_scans(args[0], opts)
