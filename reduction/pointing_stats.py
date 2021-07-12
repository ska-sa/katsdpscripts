import sys
import optparse
import logging
import time
import numpy as np
import katpoint
from katpoint import rad2deg, deg2rad
#from katsdpscripts import git_info
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger('pointing_stats')
#logger.setLevel(logging.DEBUG)


def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

def save_pointingmodel(filebase,model):
    # Save pointing model to file
    outfile = open(filebase + '.csv', 'w')
    outfile.write(model.description)
    outfile.close()
    #logger.debug("Saved %d-parameter pointing model to '%s'" % (len(model.params), filebase + '.csv'))

def read_offsetfile(filename):
    """Load data file in one shot as an array of strings."""
    string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
    data = np.loadtxt(filename, dtype=str, comments='#', delimiter=', ')
    # Interpret first non-comment line as header
    fields = data[0].tolist()
    # By default, all fields are assumed to contain floats
    formats = np.tile(float, len(fields))
    # The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
    formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
    # Convert to heterogeneous record array
    data = np.rec.fromarrays(data[1:].transpose(), dtype=list(zip(fields, formats)))
    # Load antenna description string from first line of file and construct antenna object from it
    antenna = katpoint.Antenna(open(filename,'r').readline().strip().partition('=')[2])
    # Use the pointing model contained in antenna object as the old model (if not overridden by file)
    # If the antenna has no model specified, a default null model will be used
    return data

def metrics(model,az,el,measured_delta_az, measured_delta_el ,std_delta_az ,std_delta_el):
    """Determine new residuals and sky RMS from pointing model."""
    model_delta_az, model_delta_el = model.offset(az, el)
    residual_az = measured_delta_az - model_delta_az
    residual_el = measured_delta_el - model_delta_el
    residual_xel  = residual_az * np.cos(el)
    abs_sky_error = rad2deg(np.sqrt(residual_xel ** 2 + residual_el ** 2)) * 69. 
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
    chi2 = np.sum(((residual_xel / std_delta_az) ** 2 + (residual_el / std_delta_el) ** 2))
    text = []
    #text.append("$\chi^2$ = %g " % chi2)
    text.append("All sky RMS = %.3f\" (robust %.3f\") " % (sky_rms, robust_sky_rms))
    return sky_rms,robust_sky_rms,chi2,text

def param_to_str(model, p):
    """Represent value of *p*'th parameter of *model* as a string."""
    parameter = [param for param in model][p]
    # Represent P9 and P12 (scale parameters) in shorter form
    return parameter.value_str if p not in [8, 11] else ("%.3e" % parameter.value)



string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
now = time.strftime('%Y-%m-%d_%Hh%M') # current time stamp
parser = optparse.OptionParser(usage="%prog [options] <data  files > ",
                               description="This checkes the fit of the pointing model to the given data CSV file"
                               "This will also return the robust RMS for each antenna.")
parser.add_option('-c', '--compare', action='store_false', default=True,
                  help="Do not plot comparison between fit models.")
parser.add_option('-n', '--no-stats', dest='use_stats', action='store_false', default=True,
                  help="Ignore uncertainties of data points during fitting")
parser.add_option('-o', '--output', dest='outfilebase', default='pointing_model_%s' % (now,),
                  help="Base name of output files (*.csv for new pointing model and *_data.csv for residuals, "
                  "default is 'pointing_model_<time>')")
parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False,
                  help="Print more output.")
parser.add_option('-r', '--reduced', dest='reduced', action='store_true', default=False,
                  help="Print minimum output")

parser.add_option('-p', '--pointing-model', dest='pmfilename',
                  help="Name of optional file containing old pointing model (overrides the usual one in CSV file)")
# Minimum pointing uncertainty is arbitrarily set to 1e-12 degrees, which corresponds to a maximum error
# of about 10 nano-arcseconds, as the least-squares solver does not like zero uncertainty
parser.add_option('-m', '--min-rms', type='float', default=np.sqrt(2) * 60. * 1e-12,
                  help="Minimum uncertainty of data points, expressed as the sky RMS in arcminutes")
(opts, args) = parser.parse_args()
min_rms=opts.min_rms
text = []

##########################
# Get data from files.
##########################
result = {}
for arg in args: 
    if len(args) < 1 or not arg.endswith('.csv'):
        raise RuntimeError('Correct File not passed to program. File should be csv file')
    filename = arg
    data = read_offsetfile(filename)
    keep = np.ones((len(data)),dtype=bool)

    #####################################
    # Load old pointing model, if given.
    #####################################
    old_model = None
    if opts.pmfilename:
        try:
            old_model = katpoint.PointingModel(open(opts.pmfilename).readline())
            logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(old_model), opts.pmfilename))
        except IOError:
            logger.warning("Could not load old pointing model from '%s'" % (opts.pmfilename,))

    # If the antenna has no model specified, a default null model will be used
    antenna = katpoint.Antenna(open(filename).readline().strip().partition('=')[2])
    if old_model is None:
        old_model = antenna.pointing_model

    #keep = data['keep'].astype(bool) if 'keep' in data.dtype.fields else np.tile(True, len(targets))

    ##########################################
    # Initialise new pointing model and set 
    # default enabled parameters.
    ##########################################
    num_params = len(old_model)
    default_enabled = np.array([1, 3, 4, 5, 6, 7,8]) - 1
    enabled_params = np.tile(False, num_params)
    enabled_params[default_enabled] = True
    enabled_params = enabled_params.tolist()


    # Fit new pointing model
    az, el = angle_wrap(deg2rad(data['azimuth'])), deg2rad(data['elevation'])
    measured_delta_az, measured_delta_el = deg2rad(data['delta_azimuth']), deg2rad(data['delta_elevation'])
    # Uncertainties are optional
    min_std = deg2rad(min_rms  / 60. / np.sqrt(2))
    std_delta_az = np.clip(deg2rad(data['delta_azimuth_std']), min_std, np.inf) \
        if 'delta_azimuth_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(az))
    std_delta_el = np.clip(deg2rad(data['delta_elevation_std']), min_std, np.inf) \
        if 'delta_elevation_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(el))


    ###############################################
    # Perform calculations and assign text output.
    ###############################################
    text.append("Blind Pointing metrics for %s N = %i  test Data Points "%(antenna.name,np.sum(keep)))

    sky_rms,robust_sky_rms,chi2,text1 = metrics(old_model,az[keep],el[keep],measured_delta_az[keep], measured_delta_el[keep] ,std_delta_az[keep] ,std_delta_el[keep])
    text += text1
    text.append('')
    text.append('')
    i = 0
    tmpstr = ""
    #print new_model.description
    if opts.verbose :
        for line in text: print(line)
    result[antenna.name] = robust_sky_rms

for i in range(64):
    ant = "m%03i"%(i)
    if ant not in result.keys():
        result[ant]=''
    
if not opts.reduced:
    print("Date , ", data['timestamp_ut'][0])
else: 
    print(data['timestamp_ut'][0])   
band = 'L'
if data['frequency'][0] < 1.2e9 :
    band = 'U'
if data['frequency'][0] > 1.7e9 :
    band = 'S'

if not opts.reduced:
    print("Band , ", band)
else:
    print(band)   
    
for ant in sorted(result):
    if not opts.reduced:
        print("%s , %s"%(ant,result[ant]))
    else:
        print("%s"%(result[ant]))
        
