#!/usr/bin/env python
# getH5sensors.py , last mod 01/02/2016 [NJY]

#########################################################
# Print/plot sensor and log information for hdf5 files.
##########################################################

import sys, glob, time, os, katdal
from optparse import OptionParser   # enables variables to be parsed from command line
import numpy as np
import matplotlib.pyplot as p
from matplotlib.ticker import AutoMinorLocator

###########################################
# Define useful functions for general use.
###########################################
def get_files(parentDir):
    """Get list of files in subdirectories when using recursive file search."""
    all_files = np.array([],dtype=str)
    subDirs = [d[0] for d in os.walk(parentDir)]
    files = [np.array(glob.glob(parentDir+'/*h5')) for parentDir in subDirs]
    for f in files:
        if ( f.shape[0] > 0 ):
            all_files = np.append(all_files,f) 
    return all_files

def get_data(file,vSwitch):
    """Read hdf5 data and obtain relevant metadata information."""
    h5 = katdal.open(file)
    obs = h5.description
    sensors = h5.sensor.keys()
    tvals = h5.timestamps - h5.timestamps[0] # get updated timestamps (s)
    descr = '\n'.join(h5.__str__().split('\n')[:23]) # human-friendly header information
    if ( vSwitch == 1 ):
        print '\n', descr # print header
    return h5, obs, tvals, sensors

def plot_data(x,y,yLabel):
    """Plot sensor data."""
    fig = p.figure(num=1, figsize=(13, 8.9), dpi=80)
    p.clf()
    params = {'axes.labelsize': 18, 'font.size': 15, 'legend.fontsize': 10, 
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'text.usetex': True}
    p.rcParams.update(params)

    ax = fig.add_subplot(111)
    p.plot(x,y,'b-')
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    p.ylabel(str(yLabel))
    p.xlabel('Time (s)')
    p.xlim(x.min(),x.max())
    return 0

##############################################
# Initialise parameters from command line.
##############################################
parser = OptionParser(usage="%prog <options>", description='Print/plot sensor information for hdf5 data.')
parser.add_option("-d", "--dir", default=None, type='string', help="Directory to look into for data.")
parser.add_option("-f", "--files", default=None, type='string', help="Input file override (default = ./*h5).")
parser.add_option("-o", "--outfile", default=None, type='string', help="Ouput file override (default = None.)")
parser.add_option("-p", "--plot", default=False, action='store_true', help="Plot sensor data (default = False).")
parser.add_option("-r", "--recursive", default=False, action='store_true', help="Recursively search list of directories (default = False.)")
parser.add_option("-s", "--sensor", default=False, action='store_true', help="Retrieve sensor information (default = False.)")
parser.add_option("-v", "--verbose", default=False, action='store_true', help="Print verbose information (default = False.)")
(opts, args) = parser.parse_args()
t0 = time.time()  # record script start time

##############################################
# Populate file list from working directory.
##############################################
if ( opts.dir == '.' ): opts.dir = './'
if ( opts.files != None ):
    if ( opts.dir != None ):
        files = [opts.dir + f for f in opts.files.split()]
    else:
        files = opts.files.split()
else:
    if ( opts.dir != None ):
        if ( opts.recursive is True ):
            files = get_files(opts.dir)
        else:
            files = np.array(glob.glob(opts.dir+"/*.h5"))
        if ( files.shape[0] > 1 ):
            print '\n List of files to choose from:\n', files
            print '\n Select file to analyse or press enter to select all...'
            file = raw_input()
            if ( file != '' ):
                files = file
        else:
            if ( files.shape[0] == 0 ):
                print '\n Directory specified does not contain hdf5 files. Exiting now...\n'
                sys.exit(1)
            elif ( files.shape[0] == 1 ):
                print '\n File selected = %s' %files[0]            
    else:
        print '\n Specify file(s) or directory with file(s) to analyse. Exiting now...\n'
        sys.exit(1)

####################################
# Get metadata from hdf5 files.
####################################
index = 0 
badFiles = np.array([],dtype=str)
if ( opts.outfile != None ): f_handle = open(opts.outfile,'wa+')
for file in files:
    try:
        h5, obs, tvals, sensors = get_data(file,opts.verbose)
    except(IOError):
        badFiles = np.append(badFiles,file)
        continue

    try:
        log = h5.obs_script_log
    except(IndexError,KeyError):
        badFiles = np.append(badFiles,file)
    
    try:
        val = float(log[0]) 
        if ( np.isnan(val) == True ):
            badFiles = np.append(badFiles,file)
    except(ValueError,TypeError):
        pass
    
    if ( opts.verbose is False ):
        if ( opts.outfile != None ):
            f_handle.write(('%s -> %s\n') %(file,obs))
            #np.savetxt(f_handle,np.c_[file,obs],fmt='%s -> %s')
        else:
            print ' %s -> %s' %(file,obs)
    if ( index == 0 ) and ( opts.sensor is True ):
        print '\n Full list of sensor information:\n',  ' \n'.join(sensors), '\n'
        sensor = raw_input()
    
    if ( opts.sensor is True ):
        try:
            p.ion()
            sensorData =  h5.sensor[sensor]
            plot_data(tvals,sensorData,sensor.replace('_','-'))                
            p.show()
            print ' Press enter to continue...'
            var = raw_input()
        except(IndexError,ValueError,TypeError):
            badFiles = np.append(badFiles,file)
    index += 1

if ( opts.outfile != None ):
    f_handle.write(('\n List of files with broken script logs/IOerrors:\n%s') %('\n'.join(badFiles.tolist())))
    f_handle.close()
else:
    print '\n List of files with broken script logs/ IOErrors:\n', badFiles, '\n'

#*********************************#
t1 = time.time() # record script finish time
print ' Total elapsed time: %.2f s (%.2f mins)\n' %(t1-t0,(t1-t0)/60.)