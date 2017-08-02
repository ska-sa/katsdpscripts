#!/usr/bin/env python

# Copyright (C) 2017 by Maciej Serylak
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform recipients that you have modified the original work.

import os
import sys
import glob
import argparse
import subprocess
import time
import numpy as np
from matplotlib import cm
from matplotlib import colors
import psrchive
from coast_guard import cleaners
from coast_guard import clean_utils
from coast_guard import utils


def get_archive_info(archive):
    '''Query archive attributes.
       Input:
          archive: loaded PSRCHIVE archive object.
       Output:
           Print attributes of the archive.
    '''
    file_name = archive.get_filename()
    nbin = archive.get_nbin()
    nchan = archive.get_nchan()
    npol = archive.get_npol()
    nsubint = archive.get_nsubint()
    obs_type = archive.get_type()
    telescope_name = archive.get_telescope()
    source_name = archive.get_source()
    ra = archive.get_coordinates().ra()
    dec = archive.get_coordinates().dec()
    centre_frequency = archive.get_centre_frequency()
    bandwidth = archive.get_bandwidth()
    DM = archive.get_dispersion_measure()
    RM = archive.get_rotation_measure()
    is_dedispersed = archive.get_dedispersed()
    is_faraday_rotated = archive.get_faraday_corrected()
    is_pol_calib = archive.get_poln_calibrated()
    data_units = archive.get_scale()
    data_state = archive.get_state()
    obs_duration = archive.integration_length()
    obs_start = archive.start_time().fracday() + archive.start_time().intday()
    obs_end = archive.end_time().fracday() + archive.end_time().intday()
    receiver_name = archive.get_receiver_name()
    receptor_basis = archive.get_basis()
    backend_name = archive.get_backend_name()
    backend_delay = archive.get_backend_delay()
    low_freq = archive.get_centre_frequency() - archive.get_bandwidth() / 2.0
    high_freq = archive.get_centre_frequency() + archive.get_bandwidth() / 2.0
    print '\nfile             Name of the file                           %s' % file_name
    print 'nbin             Number of pulse phase bins                 %s' % nbin
    print 'nchan            Number of frequency channels               %s' % nchan
    print 'npol             Number of polarizations                    %s' % npol
    print 'nsubint          Number of sub-integrations                 %s' % nsubint
    print 'type             Observation type                           %s' % obs_type
    print 'site             Telescope name                             %s' % telescope_name
    print 'name             Source name                                %s' % source_name
    print 'coord            Source coordinates                         %s%s' % (ra.getHMS(), dec.getDMS())
    print 'freq             Centre frequency (MHz)                     %s' % centre_frequency
    print 'bw               Bandwidth (MHz)                            %s' % bandwidth
    print 'dm               Dispersion measure (pc/cm^3)               %s' % DM
    print 'rm               Rotation measure (rad/m^2)                 %s' % RM
    print 'dmc              Dispersion corrected                       %s' % is_dedispersed
    print 'rmc              Faraday Rotation corrected                 %s' % is_faraday_rotated
    print 'polc             Polarization calibrated                    %s' % is_pol_calib
    print 'scale            Data units                                 %s' % data_units
    print 'state            Data state                                 %s' % data_state
    print 'length           Observation duration (s)                   %s' % obs_duration
    print 'start            Observation start (MJD)                    %.10f' % obs_start
    print 'end              Observation end (MJD)                      %.10f' % obs_end
    print 'rcvr:name        Receiver name                              %s' % receiver_name
    print 'rcvr:basis       Basis of receptors                         %s' % receptor_basis
    print 'be:name          Name of the backend instrument             %s' % backend_name
    print 'be:delay         Backend propn delay from digi. input.      %s\n' % backend_delay


def get_zero_weights(archive, psrsh, verbose = False):
    '''Query the number of subint-channels with zeroed
       weights (i.e. cleaned) in TimerArchive/PSRFITS file.
       Input:
           archive: loaded PSRCHIVE archive object.
           psrsh: name of psrsh file
           verbose: verbosity flag
       Output:
           Writes out psrsh file with zap commands.
     '''
    weights = archive.get_weights()
    (nsubint, nchan) = weights.shape
    if verbose:
        print '%s has %s subints and %s channels.' % (archive.get_filename(), nsubint, nchan)
    psrsh_file = open(psrsh, 'w')
    psrsh_file.write('#!/usr/bin/env psrsh\n\n')
    psrsh_file.write('# Run with psrsh -e <ext> <script>.psh <archive>.ar\n\n')
    i = j = counter = spectrum = 0
    empty_channels = [i for i in range(nchan)]
    for i in range(nsubint):
        spectrum = 0
        del empty_channels[:]
        for j in range(nchan):
            if weights[i][j] == 0.0:
                counter += 1
                spectrum += 1
                empty_channels.append(j)
        if verbose:
            percent_subint = (float(spectrum) / float(nchan)) * 100
            print 'Subint %s has %s channels (%.2f%%) set to zero. %s' % (i, spectrum, percent_subint, empty_channels)
        for k in range(len(empty_channels)):
            #print 'zap such %d,%d' % (i, empty_channels[k])
            np.savetxt(psrsh_file, np.c_[i, empty_channels[k]], fmt='zap such %d,%d')
    total_percent = (float(counter)/float(weights.size)) * 100
    if verbose:
      print '%s data points out of %s with weights set to zero.' % (counter, weights.size)
      print '%.2f%% data points set to zero.' % (total_percent)


def replace_nth(string, source, target, position):
    '''Replace N-th occurrence of a sub-string in a string.
       TODO: add error if no instance found.
       Input:
           string: string to search in.
           source: sub-string to replace.
           target: sub-string to replace with.
           position: nth occurrence of source to replace.
       Output:
           String with replaced sub-string.
    '''
    indices = [i for i in range(len(string) - len(source) + 1) if string[i:i + len(source)] == source]
    if len(indices) < position:
        # or maybe raise an error
        return
    # can't assign to string slices. So, let's listify
    string = list(string)
    # do position-1 because we start from the first occurrence of the string, not the 0-th
    string[indices[position - 1]:indices[position - 1] + len(source)] = target
    return ''.join(string)


# Main body of the script.
if __name__ == '__main__':
    # Parsing the command line options.
    parser = argparse.ArgumentParser(usage = 'reduce_meerkat.py --indir=<input_dir> [options]',
                                     description = 'Reduce MeerKAT TimerArchive/PSRFITS data.',
                                     formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 100, width = 250),
                                     epilog = 'Copyright (C) 2017 by Maciej Serylak')
    parser.add_argument('--indir', dest = 'input_dir', metavar = '<input_dir>', default = '', help = 'specify input directory')
    parser.add_argument('--outdir', dest = 'output_dir', metavar = '<output_dir>', default = '', help = 'specify output directory')
    parser.add_argument('--eph', dest = 'ephem_file', metavar = '<ephem_file>', default = '', help = 'use ephemeris file to update archives')
    parser.add_argument('--psrsh', dest = 'psrsh_save', action = 'store_true', help = 'write zap commands to psrsh script file')
    parser.add_argument('--clean', dest = 'clean_rfi', action = 'store_true', help = 'clean data from RFI using CoastGuard\'s clean.py')
    parser.add_argument('--fscr', dest = 'fscr', action = 'store_true', help = 'dedisperse, frequency scrunch and write out the file')
    parser.add_argument('--tscr', dest = 'tscr', action = 'store_true', help = 'dedisperse, time scrunch and write out the file')
    parser.add_argument('--ntscr', dest = 'tscr_nsub', nargs = 1, help = 'dedisperse, time scrunch to n-subints and write out the file')
    parser.add_argument('--verbose', dest = 'verbose', action = 'store_true', help = 'print debugging information')
    args = parser.parse_args() # Reading command line options.

    # Start timing the script.
    script_start_time = time.time()

    # Check for input_dir presence.
    if not args.input_dir:
        print parser.description, '\n'
        print 'Usage:', parser.usage, '\n'
        print parser.epilog
        sys.exit(0)

    # Validate writing permissions of input_dir.
    if os.access(args.input_dir, os.W_OK):
        pass
    else:
        print '\nInput directory without write permissions. Exiting script.\n'
        sys.exit(0)

    # Check for output_dir presence and validate writing permissions.
    if not args.output_dir:
        print '\nOption --outdir not specified. Selecting input directory.\n'
        output_dir = args.input_dir
    else:
        if os.access(args.output_dir, os.F_OK):
            output_dir = args.output_dir
        else:
            print '\nOutput directory does not exist. Exiting script.\n'
            sys.exit(0)
        if os.access(args.output_dir, os.W_OK):
            pass
        else:
            print '\nOutput directory without write permissions. Exiting script.\n'
            sys.exit(0)

    # Read ephemeris and check if it conforms to standard (has EPHVER in it).
    # This is very rudimentary check. One should check for minimal set of values
    # present in the par file. More through checks are on TODO list.
    ephem_file = args.ephem_file
    if not ephem_file:
        if args.verbose:
            print '\nOption --eph not specified. Continuing without updating ephemeris.\n'
        update_ephem = False
    else:
        if 'EPHVER' not in open(ephem_file).read():
            print '\nProvided file does not conform to ephemeris standard. Exiting script.\n'
            sys.exit(0)
        else:
            update_ephem = True

    # Check for TimerArchive/PSRFITS files and add them together.
    input_files = []
    input_dir = args.input_dir + '/'
    for file in os.listdir(input_dir):
        if file.endswith('.ar'):
            input_files.append(file)
    input_files.sort()
    if len(input_files) < 1:
        print '\nFound no matching TimerArchive/PSRFITS files. Exiting script.\n'
        sys.exit(0)
    archives = []
    archives = [psrchive.Archive_load(input_dir + file) for file in input_files]
    if args.verbose:
        print '\nLoading TimerArchive/PSRFITS files:'
    for i in range(1, len(archives)):
        if args.verbose:
            print archives[i]
        archives[0].append(archives[i])

    # Prepare new data object and set filename stem (PSR_DYYYYMMDDTHHMMSS convention).
    raw_archive = archives[0].clone()
    if args.verbose:
        print '\nFile attributes:'
        get_archive_info(raw_archive)
    filename_stem = raw_archive.get_source() + '_D' + replace_nth(os.path.split(raw_archive.get_filename())[-1], '-', 'T', 3).replace('-','').replace(':','')[:-3]
    if args.verbose:
        print '\nUsing filename stem: %s\n' % filename_stem
    raw_archive.unload(output_dir + '/' + filename_stem + '.ar')
    raw_archive = psrchive.Archive_load(output_dir + '/' + filename_stem + '.ar')

    # Update ephemeris and write out the file.
    if update_ephem:
        print '\nUpdating ephemeris in: %s\n' % raw_archive.get_filename()
        raw_archive.set_ephemeris(ephem_file)

    # Clean archive from RFI and save zap commands to psrsh file.
    if args.clean_rfi:
        cleaner = cleaners.load_cleaner('surgical')
        surgical_parameters = 'chan_numpieces=1,subint_numpieces=1,chanthresh=3,subintthresh=3'
        cleaner.parse_config_string(surgical_parameters)
        print '\nCleaning archive from RFI.\n'
        cleaner.run(raw_archive)
        if args.psrsh_save:
            psrsh_filename = output_dir + '/' + filename_stem + '.ar.psh'
            if args.verbose:
                print '\nSaving zap commands to psrsh script: %s\n' % psrsh_filename
            get_zero_weights(raw_archive, psrsh_filename)
        print '\nSaving data to file %s\n' % (filename_stem + '.ar.zap')
        raw_archive.unload(output_dir + '/' + filename_stem + '.ar.zap')

    # Prepare de-dispersed and freq. resolved average profile.
    if args.tscr:
        tscrunch_archive = raw_archive.clone()
        tscrunch_archive.dedisperse()
        tscrunch_archive.tscrunch()
        if args.clean_rfi:
            tscrunch_archive.unload(output_dir + '/' + filename_stem + '.ar.zap.DT')
        else:
            tscrunch_archive.unload(output_dir + '/' + filename_stem + '.ar.DT')

    # Prepare de-dispersed and time resolved average profile.
    if args.fscr:
        raw_archive.dedisperse()
        raw_archive.fscrunch()
        if args.clean_rfi:
            raw_archive.unload(output_dir + '/' + filename_stem + '.ar.zap.DF')
        else:
            raw_archive.unload(output_dir + '/' + filename_stem + '.ar.DF')

    if args.tscr_nsub:
        raw_archive.tscrunch_to_nsub(int(args.tscr_nsub[0]))
        if args.clean_rfi:
            raw_archive.unload(output_dir + '/' + filename_stem + '.ar.zap.DF.T' + args.tscr_nsub[0])
        else:
            raw_archive.unload(output_dir + '/' + filename_stem + '.ar.DF.T' + args.tscr_nsub[0])

    # End timing the script and output running time.
    script_end_time = time.time()
    print '\nScript running time: %.1f s.\n' % (script_end_time - script_start_time)
