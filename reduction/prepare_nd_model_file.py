###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
#! /usr/bin/python
#
# Convert noise diode measurements from contractors to internal format
#

import optparse
import sys
import re

parser = optparse.OptionParser(usage="%prog [options] file",
                               description='Convert EMSS cryostat measurement file to pair '
                                           'of KAT CSV config files, one per polarisation')

parser.add_option('-a', '--antenna', help="Antenna containing this cryostat assembly, as e.g. 'ant1', *required*")
parser.add_option('-d', '--diode', type='choice', choices=('coupler', 'pin'),
                  help="Diode tested in this file ('coupler' or 'pin'), *required*")
# Placeholder for polarisation (set automatically for both H and V output files, don't set this yourself!)
parser.add_option('--pol', help=optparse.SUPPRESS_HELP)
parser.add_option('-i', '--interp', help='Scape interpolation function to use (default is linear)')
parser.add_option('-t', '--date', help='Date of measurements (could be approximate)')
parser.add_option('-c', '--cryostat', help='Cryostat assembly number according to EMSS')
parser.add_option('-n', '--doc', help='KAT eB document number')
parser.add_option('-r', '--rev', help='KAT eB document revision')
parser.add_option('-x', '--notes', help='Extra notes')

parser.set_defaults(interp='PiecewisePolynomial1DFit(max_degree=1)')
# Parse the command line and check options and arguments
opts, args = parser.parse_args()
if len(args) != 1:
    print('Please specify a single input measurement filename as argument')
    sys.exit(1)
if opts.antenna is None:
    print("Please specify the antenna for this cryostat (e.g. 'ant1')")
    sys.exit(1)
if opts.diode is None:
    print("Please specify the noise diode characterised in this file ('coupler' or 'pin')")
    sys.exit(1)

# Load data as strings and parse them into columns
lines = file(args[0]).readlines()
three_columns = re.compile('\A\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\Z')
assert three_columns.match(lines[0]) is None, "First line of EMSS file is expected to be part of header"
assert three_columns.match(lines[1]) is None, "Second line of EMSS file is expected to be part of header"
try:
    data = [three_columns.match(line).groups() for line in lines[2:]]
except AttributeError:
    raise AssertionError('EMSS file is expected to contain three space-separated columns of numbers')

# Extract output header keys from option list
keys = [opt.dest for opt in parser.option_list if opt.dest is not None]

# Create separate files for each polarisation
for pol in ('H', 'V'):
    outfile = file('%s.%s.%s.csv' % (opts.antenna, opts.diode, pol.lower()), 'w')
    opts.pol = pol
    # Write key-value header if specified in options
    for key in keys:
        val = getattr(opts, key, None)
        if val is not None:
            outfile.write('# %s = %s\n' % (key, val))
    outfile.write('#\n# Frequency [Hz], Temperature [K]\n')
    # Write CSV part of file
    outfile.write(''.join(['%se9, %s\n' % (entry[0], entry[1] if pol == 'H' else entry[2]) for entry in data]))
    outfile.close()
