#!/usr/bin/python
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################


from __future__ import with_statement
import arutils
from matplotlib.widgets import RectangleSelector, Button
import matplotlib.pyplot as plt
import sys, optparse, glob, logging
import os.path, katpoint
import numpy as np
from scape import DataSet, extract_xyz_data

# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <directories or files>",
                               description="This processes one or more datasets (HDF5) and extracts RFI affected channels from them.")
parser.add_option('-a', '--baseline', default='sd',
                help="Baseline to load (e.g. 'A1A1' for antenna 1), default is first single-dish baseline in file")
parser.add_option("-f", "--frequency_channels", dest="freq_keep", default='70,460',
                help="Range of frequency channels to keep (zero-based, specified as start,end). Default = %default")
parser.add_option("-o", "--output", dest="outfilebase", default='pre-defined_rfi',
                help="Base name of output files (*.csv for output data and *.log for messages)")
parser.add_option("-c", "--output_chan", dest="outfilebase2", default='contaminte_rfi_channels',
                help="Base name of output files (*.txt for output data and *.log for messages)")
parser.add_option("-s", "--size", dest="file_size", type='float', default=100.0,
                help="Size of the file to be reduced Default = %default")
parser.add_option("-m", "--startdate", default='01/05/2010',
                    help="filtered start date (e.g day/month/year) for the datasets to be reduced")
parser.add_option("-e", "--enddate", default='15/05/2010',
                    help="filtered end date (e.g day/month/year ) for the data set to be reduced")
parser.add_option("-n", "--ant", default='1',
                help="filtered first antenna from the data set to be reduced")
(opts, args) = parser.parse_args()
if len(args) < 1:
    args = ['.']

# Set up logging: logging everything (DEBUG & above), both to console and file
logger = logging.root

# Find all data sets (HDF5)
datasets, output_data = [],[]
output_chan, output_ts, abs_time = [], [], []
output_az, output_el = [], []

#ar = arutils.ArchiveBrowser(arutils.karoo_archive_cpt)
#ar.filter_by_date(start_date=opts.startdate, end_date=opts.enddate)
#ar._filter_from_string('antennas', '^%s' % opts.ant)
#datasets = [i['file_name'] for i in ar.kath5s if os.path.getsize(i['file_name'])/1e6 < opts.file_size]
#datasets = [i['file_name'] for i in ar.kath5s]
#print "Length of datasets found is:", len(datasets)

def walk_callback(arg, directory, files):
    datasets.extend([os.path.join(directory, f) for f in files if f.endswith('.h5') and os.path.getsize(os.path.join(directory, f))/1e6 <= opts.file_size])
for arg in args:
    if os.path.isdir(arg):
        os.path.walk(arg, walk_callback, None)
    else:
        datasets.extend(glob.glob(arg))
if len(datasets) == 0:
    raise ValueError('No data sets (HDF5) found')
else:
    print "LENNGTH OF DATASET FOUND IS:",len(datasets),"\nDATASETS ARE:", datasets

FIRST_INDEX = 0

# Figure
fig = plt.figure()
current_ax = plt.subplot(111, axisbg='#FFFFCC')
plt.subplots_adjust(bottom=0.2)
current_ax.set_xlabel('Frequency Channels [MHz]', bbox=dict(facecolor='red'))
current_ax.set_ylabel('Power [Count]', bbox=dict(facecolor='red'))
current_ax.set_title("CLICK NEXT TO LOAD THE DATA SET")

print "CLICK NEXT TO LOAD THE FIRST DATA SET TO BE REDUCED"

# Indices to step through data sets as the buttons are pressed
class Index:
    def __init__(self):
        self.ind = FIRST_INDEX
    def next(self, event):
        azimuth, elevation = [], []
        # Opening the out put file to write the RFI contaminated channels information
        if self.ind >= len(datasets):
            print "No more data to be reduced, hold on a second while we writting the extracted info in the file."
            fout = file(opts.outfilebase + '.csv', 'w')
            fout.write("FILENAME, FREQUCENCY[MHz], TIMESTAMPS, ABS_TIME, AZIMUTH, ELEVATION, PEAK POWER [dB]\n")
            fout.writelines([('%s, %0.4f, %0.4f, %s, %0.2f, %0.2f,%0.2f \n') % tuple(p) for p in output_data if p])
            fout.close()

            # Opening the out put file to write only RFI conteminated channels
            fout2 = file(opts.outfilebase2 + '.txt', 'w')
            fout2.write("CONTAMINATED FREQUCENCY CHANNELS [MHz]\n")
            fout2.write("======================================\n")
            fout2.writelines([('%0.4f\n') % p for p in set(output_chan) if p])
            fout2.close()

            # Time vs Frequency for selected channels figure
            fig_new = plt.figure()
            new_ax1 = fig_new.add_subplot(311,axisbg='#FFFFCC' )
            new_ax1.plot(output_ts, output_chan,'k+',lw=3)
            new_ax1.set_xlabel("Time [s]")
            new_ax1.set_ylabel("Frequency [MHz]")
            new_ax2 = fig_new.add_subplot(312, axisbg="#FFFFCC")
            new_ax2.plot(output_az,output_chan,'g+', lw=3)
            new_ax2.set_xlabel("Azimuth [Deg]")
            new_ax2.set_ylabel("Frequency [MHz]")
            new_ax3 = fig_new.add_subplot(313, axisbg='#FFFFCC')
            new_ax3.plot(output_az, output_el, 'r+', lw=3)
            new_ax3.set_xlabel("Azimuth [Deg]")
            new_ax3.set_ylabel("Elevation [Deg]")
            print "We done writting in to a file,look at the frequency vs time plot to see the RFI mapping as a function of time"
            print "The RFI contaminated channels are:", set(output_chan)
            plt.show()
            sys.exit()

        self.filename = datasets[self.ind]
        try:
            #logger.info("Loading dataset %s , File size is %fMB, This is File number %s" % (os.path.basename(self.filename),os.path.getsize(self.filename)/1e6,self.ind))
            logger.info("Loading dataset %s , File size is %fMB, This is File number %s" % (os.path.basename(self.filename),os.path.getsize(self.filename),self.ind))
            current_dataset = DataSet(self.filename, baseline=opts.baseline)
            out_filename =os.path.basename(self.filename)
            start_freq_channel = int(opts.freq_keep.split(',')[0])
            end_freq_channel = int(opts.freq_keep.split(',')[1])
            current_dataset = current_dataset.select(freqkeep=range(start_freq_channel, end_freq_channel+1))
            current_dataset = current_dataset.select(labelkeep='scan', copy=False)
            if len(current_dataset.compscans) == 0 or len(current_dataset.scans) == 0:
                logger.warning('No scans found in file, skipping data set')
         
            # try to extract antenna target points per each timestamps
            for cscan in current_dataset.compscans:
                target = cscan.target.name
                az = np.hstack([scan.pointing['az'] for scan in cscan.scans])
                el = np.hstack([scan.pointing['el'] for scan in cscan.scans])
                ts = np.hstack([scan.timestamps for scan in cscan.scans])
                azimuth.extend(katpoint.rad2deg(az)),elevation.extend(katpoint.rad2deg(el))

            azimuth, elevation = np.array(azimuth), np.array(elevation)
            ts,f,amp = extract_xyz_data(current_dataset,'abs_time','freq','amp')
            power,freq = amp.data,f.data
            t = np.hstack(ts.data)
            base_freq = freq[0]
            p = np.hstack(power)
            T,F = np.meshgrid(t,base_freq)
            A,F = np.meshgrid(azimuth,base_freq)
            E,F = np.meshgrid(elevation,base_freq)
            AA = A.ravel()
            EE = E.ravel()
            TT = T.ravel()
            FF = F.ravel()
            PP = p.ravel()
            
            def onselect_next(eclick,erelease):
                global output_data, output_chan, output_ts
                xmin = min(eclick.xdata, erelease.xdata)
                xmax = max(eclick.xdata, erelease.xdata)
                ymin = min(eclick.ydata, erelease.ydata)
                ymax = max(eclick.ydata, erelease.ydata)

                ind = (FF >= xmin) & (FF <= xmax)  & (PP >= ymin) & (PP <= ymax)
                selected_freq = FF[ind]
                selected_amp = 10.0*np.log10(PP[ind])
                selected_ts = TT[ind]
                selected_az = AA[ind]
                selected_el = EE[ind]
                print "SUCCESSFUL, CLICK AND DRAG TO SELECT THE NEXT RFI CHANNELS OR NEXT TO LOAD NEW DATASET"

                #sorting with increasng X_new
                indices = np.lexsort(keys = (selected_ts, selected_freq))

                for index in indices:
                    output_data.append([out_filename, selected_freq[index],selected_ts[index], katpoint.Timestamp(selected_ts[index]).local(), selected_az[index], selected_el[index], selected_amp[index]])
                for point in output_data:
                    output_chan.append(point[1])
                    output_ts.append(point[2])
                    output_az.append(point[4])
                    output_el.append(point[5])
                    
            def toggle_selector_next(event):
                print ' Key pressed.'
                if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                    print ' RectangleSelector deactivated.'
                    toggle_selector_next.RS.set_active(False)
                if event.key in ['A', 'a'] and not toggle_selector_next.RS.active:
                    print ' RectangleSelector activated.'
                    toggle_selector_next.RS.set_active(True)

            # New Figure for the current data set
            current_ax.clear()
            plt.subplots_adjust(bottom=0.2)
            current_ax.plot(FF,PP, '+')
            current_ax.set_title("CLICK AND DRAG TO SELECT RFI CHAN")
            current_ax.set_xlabel('Frequency Channels [MHz]', bbox=dict(facecolor='red'))
            current_ax.set_ylabel('Power [Count]', bbox=dict(facecolor='red'))
            plt.draw()

            print "NEW DATA SET SUCCESSFLY LOADED, CLICK AND DRAG TO SELECT THE RFI CONTAMINATED CHANNELS OR NEXT TO CONTINUE"

            toggle_selector_next.RS = RectangleSelector(current_ax, onselect_next, drawtype='box')
            plt.connect('key_press_event', toggle_selector_next)
        except ValueError:
            print os.path.basename(self.filename), "DATA CORUPTED, PLEASE CLICK NEXT TO LOAD ANOTHER DATASET"
        self.ind +=1

callback = Index()
axcolor = 'lightgoldenrodyellow'
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next', color = axcolor, hovercolor='green')
bnext.on_clicked(callback.next)
plt.show()