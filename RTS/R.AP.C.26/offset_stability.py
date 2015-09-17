import optparse
import time
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from matplotlib.projections import PolarAxes
#from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import katpoint
from katpoint import  deg2rad #,rad2deg,
from katsdpscripts.RTS import git_info,get_git_path
import pandas

#from astropy.time import Time
#from matplotlib.dates import DateFormatter


def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period



def calc_rms(x):
    """
    Finds the RMS of a set of data
    """
    if np.isnan(x).sum() >= x.shape[0]+1 : return 0.0
    z = np.ma.array(data=np.nan_to_num(x),mask=np.isnan(x))
    return np.ma.sqrt(np.ma.mean((z-z.mean())**2))

def calc_rms_total(x):
    """
    Finds the RMS of a set of data
    """
    if np.isnan(x).sum() >= x.shape[0]+1 : return 0.0
    z = np.ma.array(data=np.nan_to_num(x),mask=np.isnan(x))
    return np.ma.sqrt(np.ma.mean((z-z.mean())**2))


def calc_change(x):
    """
    Finds the RMS of a set of data
    """
    if np.isnan(x).sum() >= x.shape[0]+1 : return 0.0
    z = np.ma.array(data=np.nan_to_num(x),mask=np.isnan(x))
    return z[-1] - z[0]

def calc_change_total(x):
    """
    Finds the RMS of a set of data
    """
    if np.isnan(x).sum() >= x.shape[0]+1 : return 0.0
    z = np.ma.array(data=np.nan_to_num(x),mask=np.isnan(x))
    return z[-1] - z[0]



# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
# Create a date/time string for current time
now = time.strftime('%Y-%m-%d_%Hh%M')


def read_offsetfile(filename):
    # Load data file in one shot as an array of strings
    string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
    data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
    # Interpret first non-comment line as header
    fields = data[0].tolist()
    # By default, all fields are assumed to contain floats
    formats = np.tile(np.float, len(fields))
    # The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
    formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
    # Convert to heterogeneous record array
    data = np.rec.fromarrays(data[1:].transpose(), dtype=zip(fields, formats))
    # Load antenna description string from first line of file and construct antenna object from it
    antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])
    # Use the pointing model contained in antenna object as the old model (if not overridden by file)
    # If the antenna has no model specified, a default null model will be used
    return data,antenna


parser = optparse.OptionParser(usage="%prog [options] <data  files > ",
                               description="This works out stability measures when given a data CSV file"
                               "  "
                               " "
                               " ")
# Minimum pointing uncertainty is arbitrarily set to 1e-12 degrees, which corresponds to a maximum error
# of about 10 nano-arcseconds, as the least-squares solver does not like zero uncertainty
parser.add_option('-m', '--min-rms', type='float', default=np.sqrt(2) * 60. * 1e-12,
                  help="Minimum uncertainty of data points, expressed as the sky RMS in arcminutes")
(opts, args) = parser.parse_args()

#if len(args) != 1 or not args[0].endswith('.csv'):
#    raise RuntimeError('Please specify a single CSV data file as argument to the script')


text = []


#offset_file = 'offset_scan.csv'
#filename = '1386710316_point_source_scans.csv'
#min_rms= np.sqrt(2) * 60. * 1e-12

if len(args) < 1 or not args[0].endswith('.csv'):
    raise RuntimeError('Correct File not passed to program. File should be csv file')

data = None
for filename in args:
    if data is None:
        data,ant = read_offsetfile(filename)
    else:
        data,ant = np.r_[data,read_offsetfile(filename)]

offsetdata = data



az, el = angle_wrap(deg2rad(offsetdata['azimuth'])), deg2rad(offsetdata['elevation'])
measured_delta_az, measured_delta_el = offsetdata['delta_azimuth'], offsetdata['delta_elevation']
time_stamps = np.zeros_like(az)
for i in xrange(len(az)) :
    time_stamps[i] = katpoint.Timestamp(offsetdata['timestamp_ut'][i]).secs  # Fix Timestamps 



#print new_model.description
dataset_str = '_'.join(np.unique(offsetdata['dataset']).tolist() )
nice_filename =  '%s_%s_4_hour_offset_stability'%(dataset_str ,ant.name)
pp = PdfPages(nice_filename+'.pdf')


offset_az_ts = pandas.Series((measured_delta_az*np.cos(el)-(measured_delta_az*np.cos(el)).mean()), pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
offset_el_ts = pandas.Series((measured_delta_el-(measured_delta_el).mean()), pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
offset_total_ts = pandas.Series( np.sqrt((offset_az_ts)**2+offset_el_ts**2), pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')

#(np.sqrt(change_el**2+change_az**2)).plot()
#(offset_el_ts*3600).plot()
#max_az = ((pandas.rolling_max(offset_az_ts,4*60,0,freq='60s')-pandas.rolling_min(offset_az_ts,4*60,0,freq='60s'))*3600)
#max_el = ((pandas.rolling_max(offset_el_ts,4*60,0,freq='60s')-pandas.rolling_min(offset_el_ts,4*60,0,freq='60s'))*3600)
#min_az = ((pandas.rolling_min(offset_az_ts,4*60,0,freq='60s')-pandas.rolling_min(offset_az_ts,4*60,0,freq='60s'))*3600)
#min_el = ((pandas.rolling_min(offset_el_ts,4*60,0,freq='60s')-pandas.rolling_min(offset_el_ts,4*60,0,freq='60s'))*3600)

fig = plt.figure(figsize=(10,5))
#change_total = np.sqrt(change_el**2 + change_az**2)
(offset_el_ts*3600).plot(label='Elevation',legend=True,grid=True) 
(offset_az_ts*3600).plot(label='Azimuth',legend=True,grid=True)
(offset_total_ts*3600).plot(label='Total pointing Error',legend=True,grid=True)
dataset_str = ' ,'.join(np.unique(offsetdata['dataset']).tolist() )
target_str = ' ,'.join(np.unique(offsetdata['target']).tolist() )
plt.title("Raw Offsets :   Antenna:%s Dataset:%s Target(s): %s " %(ant.name,dataset_str ,target_str ),fontsize=10)
plt.ylabel('Offset  (arc-seconds)')
plt.xlabel('Time (UTC)',fontsize=8)
plt.figtext(0.89, 0.11,git_info(get_git_path()), horizontalalignment='right',fontsize=10)

fig.savefig(pp,format='pdf')
plt.close(fig)

#offset_az_ts = pandas.Series(measured_delta_az*np.cos(el), pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
#offset_el_ts = pandas.Series(measured_delta_el, pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
#offset_total_ts = pandas.Series( np.sqrt((measured_delta_az*np.cos(el))**2+measured_delta_el**2), pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')

fig = plt.figure(figsize=(10,5))
change_el = pandas.rolling_apply(offset_el_ts,window=4*60/6.,min_periods=0,func=calc_change,freq='360s')*3600
change_az = pandas.rolling_apply(offset_az_ts,window=4*60/6.,min_periods=0,func=calc_change,freq='360s')*3600
#change_total = np.sqrt(change_el**2 + change_az**2)
change_el.plot(label='Elevation',legend=True,grid=True) 
change_az.plot(label='Azimuth',legend=True,grid=True)
#change_total.plot(label='Total change in pointing Error',legend=True,grid=True)
dataset_str = ' ,'.join(np.unique(offsetdata['dataset']).tolist() )
target_str = ' ,'.join(np.unique(offsetdata['target']).tolist() )
plt.title("4 Hour Range  :   Antenna:%s Dataset: %s  Target(s): %s " %(ant.name,dataset_str ,target_str ),fontsize=10)
plt.ylabel('4 Hour Change  (arc-seconds)')
plt.xlabel('Time (UTC)',fontsize=10)
plt.figtext(0.89, 0.11,git_info(get_git_path()), horizontalalignment='right',fontsize=10)

fig.savefig(pp,format='pdf')
plt.close(fig)


fig = plt.figure(figsize=(10,5))
mean_rms_el = pandas.rolling_apply(offset_el_ts,window=4*60/6.,min_periods=0,func=calc_rms,freq='360s')*3600
mean_rms_az = pandas.rolling_apply(offset_az_ts,window=4*60/6.,min_periods=0,func=calc_rms,freq='360s')*3600
mean_rms_total = pandas.rolling_apply(offset_total_ts,window=4*60/6.,min_periods=0,func=calc_rms_total,freq='360s')*3600
mean_rms_el.plot(label='Elevation',legend=True,grid=True) 
mean_rms_az.plot(label='Azimuth',legend=True,grid=True)
mean_rms_total.plot(label='Total pointing Error',legend=True,grid=True)
dataset_str = ' ,'.join(np.unique(offsetdata['dataset']).tolist() )
target_str = ' ,'.join(np.unique(offsetdata['target']).tolist() )
plt.title("4 hour RMS :   Antenna:%s Dataset: %s  Target(s): %s " %(ant.name,dataset_str ,target_str ),fontsize=10)
plt.ylabel('4 Hour RMS Error (arc-seconds)')
plt.xlabel('Time (UTC)',fontsize=10)
plt.hlines(25,plt.xlim()[0],plt.xlim()[1])
plt.figtext(0.89, 0.11,git_info(get_git_path()), horizontalalignment='right',fontsize=10)
minv,maxv = plt.ylim()
if maxv < 26 : maxv = 26
plt.ylim(minv,maxv)

fig.savefig(pp,format='pdf')
plt.close(fig)


fig = plt.figure(figsize=(10,5))
temperature_ts = pandas.Series(offsetdata['temperature'], pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
wind_speed_ts = pandas.Series(offsetdata['wind_speed'], pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
temperature_ts.plot(label='Surface Temperature ',legend=True,grid=True) 
wind_speed_ts.plot(label='Wind Speed (m/s)',legend=True,grid=True)
dataset_str = ' ,'.join(np.unique(offsetdata['dataset']).tolist() )
target_str = ' ,'.join(np.unique(offsetdata['target']).tolist() )
plt.title("Wind Speed & Temperature : Dataset: %s  " %(dataset_str))
plt.ylabel('')
plt.xlabel('Time (UTC)',fontsize=8)
plt.figtext(0.89, 0.11,git_info(get_git_path()), horizontalalignment='right',fontsize=10)

fig.savefig(pp,format='pdf')
plt.close(fig)

pp.close()


#TODO Tilt infomation.
#TODO Tiltsensor values in the H5 file


#    antenna, data =    (filename,outfilebase=os.path.abspath(prep_basename),baseline=opts.bline,
#                                                            ku_band=opts.ku_band,channel_mask=opts.channel_mask,freq_chans=opts.chan_range)

