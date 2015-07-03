import sys
import optparse
import logging
import time
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.projections import PolarAxes
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import katpoint
from katpoint import rad2deg, deg2rad
from katsdpscripts.RTS import git_info,get_git_path


def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period



def save_pointingmodel(filebase,model):
    # Save pointing model to file
    outfile = file(filebase + '.csv', 'w')
    outfile.write(model.description)
    outfile.close()
    logger.debug("Saved %d-parameter pointing model to '%s'" % (len(model.params), filebase + '.csv'))



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
    return data


parser = optparse.OptionParser(usage="%prog [options] <data  files > ",
                               description="This fits a pointing model to the given data CSV file"
                               " with the targets that are included in the the offset pointing csv file "
                               " "
                               " ")
parser.add_option('-o', '--output', dest='outfilebase', default='pointing_model_%s' % (now,),
                  help="Base name of output files (*.csv for new pointing model and *_data.csv for residuals, "
                  "default is 'pointing_model_<time>')")
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
        data = read_offsetfile(filename)
    else:
        data = np.r_[data,read_offsetfile(filename)]

offsetdata = data






#print new_model.description


az, el = angle_wrap(deg2rad(offsetdata['azimuth'])), deg2rad(offsetdata['elevation'])
measured_delta_az, measured_delta_el = deg2rad(offsetdata['delta_azimuth']), deg2rad(offsetdata['delta_elevation'])

def referencemetrics(measured_delta_az, measured_delta_el):
    """Determine and sky RMS from pointing model."""
    text = []
    measured_delta_xel  =  measured_delta_az* np.cos(el) # scale due to sky shape
    abs_sky_error = np.ma.array(data=measured_delta_xel,mask=False)

    for target in set(offsetdata['target']):
        keep = np.ones((len(offsetdata)),dtype=np.bool)
        for key,targetv in enumerate(offsetdata['target']):
            keep[key] = target == targetv
        abs_sky_error[keep] = rad2deg(np.sqrt((measured_delta_xel[keep]-measured_delta_xel[keep][0]) ** 2 + (measured_delta_el[keep]- measured_delta_el[keep][0])** 2)) * 60.
        abs_sky_error.mask[ keep.nonzero()[0][0]] = True # Mask the reference element
        text.append("Test Target: '%s'  Reference RMS = %.3f' (robust %.3f')  (N=%i Data Points)" % (target,np.sqrt((abs_sky_error[keep] ** 2).mean()), np.ma.median(abs_sky_error[keep]) * np.sqrt(2. / np.log(4.)),keep.sum()-1))

    ###### On the calculation of all-sky RMS #####
    # Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
    # They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
    # standard deviation of sigma
    # The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
    # The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
    # two squared Gaussian random values, each with an expected value of sigma^2.
    sky_rms = np.sqrt(np.ma.mean(abs_sky_error ** 2))
    #print abs_sky_error
    # A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
    # which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
    robust_sky_rms = np.ma.median(abs_sky_error) * np.sqrt(2. / np.log(4.))
    text.append("All Sky Reference RMS = %.3f' (robust %.3f')   (N=%i Data Points) R.T.P.4"  % (sky_rms, robust_sky_rms,abs_sky_error.count()))
    return text

text1 = referencemetrics(measured_delta_az, measured_delta_el)
text += text1

text.append("")

nice_filename =  args[0].split('/')[-1]+ '_residual_pointing_offset'
pp = PdfPages(nice_filename+'.pdf')
for line in text: print line
fig = plt.figure(None,figsize = (10,16))
plt.figtext(0.1,0.1,'\n'.join(text),fontsize=12)
plt.figtext(0.89, 0.11,git_info(get_git_path()), horizontalalignment='right',fontsize=10)
fig.savefig(pp,format='pdf')

plt.close(fig)
pp.close()



