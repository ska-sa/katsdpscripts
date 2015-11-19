import optparse
import time
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from matplotlib.projections import PolarAxes
#from matplotlib.ticker import MultipleLocator,FormatStrFormatter
#import os
import katpoint
from katpoint import  deg2rad ,rad2deg
from katsdpscripts.RTS import git_info,get_git_path
import pandas
from katsdpscripts.reduction.analyse_point_source_scans import batch_mode_analyse_point_source_scans

#from astropy.time import Time
#from matplotlib.dates import DateFormatter


def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period


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


def metrics(model,az,el,measured_delta_az, measured_delta_el ,std_delta_az ,std_delta_el,time_stamps):
    """Determine new residuals and sky RMS from pointing model."""
    model_delta_az, model_delta_el = model.offset(az, el)
    residual_az = measured_delta_az - model_delta_az
    residual_el = measured_delta_el - model_delta_el
    residual_xel  = residual_az * np.cos(el)
    abs_sky_error = rad2deg(np.sqrt(residual_xel ** 2 + residual_el ** 2))
    
    offset_az_ts = pandas.Series(rad2deg(residual_xel), pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
    offset_el_ts = pandas.Series(rad2deg(residual_el), pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
    offset_total_ts = pandas.Series( abs_sky_error, pandas.to_datetime(time_stamps, unit='s'))#.asfreq(freq='1s')
    
    ###### On the calculation of all-sky RMS #####
    # Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
    # They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
    # standard deviation of sigma
    # The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
    # The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
    # two squared Gaussian random values, each with an expected value of sigma^2.

    sky_rms = np.sqrt(np.mean(abs_sky_error ** 2))
    # A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
    # which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
    robust_sky_rms = np.median(abs_sky_error) * np.sqrt(2. / np.log(4.))
    # The chi^2 value is what is actually optimised by the least-squares fitter (evaluated on the training set)
    #chi2 = np.sum(((residual_xel / std_delta_az) ** 2 + (residual_el / std_delta_el) ** 2))
    text = 'All sky RMS = %.3f\" (robust %.3f\") ' % (sky_rms*3600, robust_sky_rms*3600)
    fig = plt.figure(figsize=(10,5))
    #change_total = np.sqrt(change_el**2 + change_az**2)
    #(offset_el_ts*3600.).plot(label='Elevation',legend=True,grid=True,style='*') 
    #(offset_az_ts*3600.).plot(label='Azimuth',legend=True,grid=True,style='*')
    (offset_total_ts*3600.).plot(label='Total pointing Error',legend=True,grid=True,style='*')
    dataset_str = ' ,'.join(np.unique(offsetdata['dataset']).tolist() )
    #target_str = ' ,'.join(np.unique(offsetdata['target']).tolist() )
    plt.title("Offset for Antenna:%s Dataset:%s  \n   %s " %(ant.name,dataset_str ,text),fontsize=10)
    plt.ylabel('Offset  (arc-seconds)')
    plt.xlabel('Time (UTC)',fontsize=8)
    plt.figtext(0.89, 0.18,git_info(get_git_path()), horizontalalignment='right',fontsize=10)
    return fig
    
    


parser = optparse.OptionParser(usage="%prog [opts] <directories or files>",
                           description="This works out stability measures results of analyse_point_source_scans.py or an h5 file")
parser.add_option("-o", "--output", dest="outfilebase", type="string", default='',
              help="Base name of output files (*.png for plots and *.csv for gain curve data)")
#parser.add_option("-p", "--polarisation", type="string", default=None, 
#              help="Polarisation to analyse, options are I, HH or VV. Default is all available.")
parser.add_option("--condition_select", type="string", default="normal", help="Flag according to atmospheric conditions (from: ideal,optimal,normal,none). Default: normal")
#parser.add_option("--csv", action="store_true", help="Input file is assumed to be csv- this overrides specified baseline")
parser.add_option("--bline", type="string", default="sd", help="Baseline to load. Default is first single dish baseline in file")
parser.add_option("--channel-mask", type="string", default='/var/kat/katsdpscripts/RTS/rfi_mask.pickle', help="Location of rfi mask pickle file specifying channels to flag")
parser.add_option("--ku-band", action="store_true", help="Force the center frequency of the input file to be Ku band")
parser.add_option("--chan-range", default='211,3896', help="Range of frequency channels to keep (zero-based, specified as 'start,end', default is 211,3896)")
parser.add_option('-m', '--min-rms', type='float', default=np.sqrt(2) * 60. * 1e-12,
                  help="Minimum uncertainty of data points, expressed as the sky RMS in arcminutes")
parser.add_option('-n', '--no-stats', dest='use_stats', action='store_false', default=True,
                  help="Ignore uncertainties of data points during fitting")

(opts, args) = parser.parse_args()
if len(args) ==0:
    raise RuntimeError('Please specify a file to process.')

min_rms=opts.min_rms
if not args[0].endswith('.csv') and not args[0].endswith('.h5'):
    raise RuntimeError('Correct File not passed to program. File should be csv file')

if opts.ku_band:
    opts.channel_mask=None

data=None
for filename in args:
    if filename.endswith('.csv') :
        if data is None:
            data,ant = read_offsetfile(filename)
        else:
            data = np.r_[data,read_offsetfile(filename)]
    if filename.endswith('.h5') : 
        if data is None:
            ant, data = batch_mode_analyse_point_source_scans(filename,outfilebase='',baseline=opts.bline,
                        ku_band=opts.ku_band,channel_mask=opts.channel_mask,freq_chans=opts.chan_range)
        else:
            ant, data_tmp = batch_mode_analyse_point_source_scans(filename,outfilebase='',baseline=opts.bline,
                        ku_band=opts.ku_band,channel_mask=opts.channel_mask,freq_chans=opts.chan_range)
            data = np.r_[data,data_tmp]

    
offsetdata = data
keep = np.ones((len(offsetdata)),dtype=np.bool)



az, el = angle_wrap(deg2rad(offsetdata['azimuth'])),deg2rad(offsetdata['elevation'])
measured_delta_az, measured_delta_el = deg2rad(offsetdata['delta_azimuth']), deg2rad(offsetdata['delta_elevation'])
time_stamps = np.zeros_like(az)
for i in xrange(len(az)) :
    time_stamps[i] = katpoint.Timestamp(offsetdata['timestamp_ut'][i]).secs  # Fix Timestamps 



#print new_model.description
dataset_str = '_'.join(np.unique(offsetdata['dataset']).tolist() )
nice_filename =  '%s_%s_pointing_error'%(dataset_str ,ant.name)
pp = PdfPages(nice_filename+'.pdf')



new_model = katpoint.PointingModel()
num_params = len(new_model)
#default_enabled = np.array([1, 3, 4, 5, 6, 7]) - 1   # first 6 params 
default_enabled = np.array([1, 7]) - 1   # only az & el offset  params 
enabled_params = np.tile(False, num_params)
enabled_params[default_enabled] = True
enabled_params = enabled_params.tolist()
# Fit new pointing model
# Uncertainties are optional
min_std = deg2rad(min_rms  / 60. / np.sqrt(2))
std_delta_az = np.clip(deg2rad(data['delta_azimuth_std']), min_std, np.inf) \
    if 'delta_azimuth_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(az))
std_delta_el = np.clip(deg2rad(data['delta_elevation_std']), min_std, np.inf) \
    if 'delta_elevation_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(el))

params, sigma_params = new_model.fit(az[keep], el[keep], measured_delta_az[keep], measured_delta_el[keep],
                                     enabled_params=enabled_params)
print params

no_off_model = katpoint.PointingModel()
params_no_off = params
#params_no_off[0]  = 0.0
#params_no_off[6]  = 0.0

no_off_model.fromlist(params_no_off)

fig = metrics(no_off_model,az[keep],el[keep],measured_delta_az[keep], measured_delta_el[keep] ,std_delta_az[keep] ,std_delta_el[keep],time_stamps[keep])
fig.savefig(pp,format='pdf')
#plt.close(fig)
pp.close()













