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
from katsdpscripts import git_info
from matplotlib.offsetbox import AnchoredText

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

def save_pointingmodel(filebase,model):
    # Save pointing model to file
    outfile = file(filebase + '.csv', 'w')
    outfile.write(model.description)
    outfile.close()
    #logger.debug("Saved %d-parameter pointing model to '%s'" % (len(model.params), filebase + '.csv'))

def read_offsetfile(filename):
    """Load data file in one shot as an array of strings."""
    string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
    data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
    # Interpret first non-comment line as header
    fields = data[0].tolist()
    # By default, all fields are assumed to contain floats
    formats = np.tile(np.float, len(fields))
    # The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
    formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
    # Convert to heterogeneous record array
    data = np.rec.fromarrays(data[1:].transpose(), dtype=list(zip(fields, formats)))
    # Load antenna description string from first line of file and construct antenna object from it
    antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])
    # Use the pointing model contained in antenna object as the old model (if not overridden by file)
    # If the antenna has no model specified, a default null model will be used
    return data

def metrics(model,az,el,measured_delta_az, measured_delta_el ,std_delta_az ,std_delta_el):
    """Determine new residuals and sky RMS from pointing model."""
    model_delta_az, model_delta_el = model.offset(az, el)
    residual_az = measured_delta_az - model_delta_az
    residual_el = measured_delta_el - model_delta_el
    residual_xel  = residual_az * np.cos(el)
    abs_sky_error = rad2deg(np.sqrt(residual_xel ** 2 + residual_el ** 2)) * 3600.
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

class PointingResults(object):
    """Calculate and store results related to given pointing model."""
    def __init__(self, model):
        self.update(model)

    def update(self, model):
        """Determine new residuals and sky RMS from pointing model."""
        model_delta_az, model_delta_el = model.offset(az, el)
        self.residual_az = measured_delta_az - model_delta_az
        self.residual_el = measured_delta_el - model_delta_el
        self.residual_xel = self.residual_az * np.cos(el)
        self.abs_sky_error = rad2deg(np.sqrt(self.residual_xel ** 2 + self.residual_el ** 2)) * 60. # arcsecs
        self.metrics(keep)

    def metrics(self, keep):
        """Determine new residuals and sky RMS from pointing model."""
        ###### On the calculation of all-sky RMS #####
        # Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
        # They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
        # standard deviation of sigma
        # The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
        # The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
        # two squared Gaussian random values, each with an expected value of sigma^2.
        self.sky_rms = np.sqrt(np.mean(self.abs_sky_error[keep] ** 2))
        # A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
        # which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
        self.robust_sky_rms = np.median(self.abs_sky_error[keep]) * np.sqrt(2. / np.log(4.))
        # The chi^2 value is what is actually optimised by the least-squares fitter (evaluated on the training set)
        self.chi2 = np.sum(((self.residual_xel / std_delta_az) ** 2 + (self.residual_el / std_delta_el) ** 2)[keep])
        self.text = []
        self.text.append("All sky RMS = %.3f' (robust %.3f') " % (sky_rms, robust_sky_rms))

def quiver_segments(delta_az, delta_el, scale):
    """Produce line segments that indicate size and direction of residuals."""
    theta1, r1 = np.pi / 2. - az, np.pi / 2. - el
    # Create line segments in Cartesian coords so that they do not change direction when *scale* changes
    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    dx = delta_az * np.cos(el) * np.cos(az) - delta_el * np.sin(az)
    dy = -delta_az * np.cos(el) * np.sin(az) - delta_el * np.cos(az)
    x2, y2 = x1 + scale * dx, y1 + scale * dy
    theta2, r2 = np.arctan2(y2, x2), np.sqrt(x2 ** 2 + y2 ** 2)
    return np.c_[np.c_[theta1, r1], np.c_[theta2, r2]].reshape(-1, 2, 2)

def param_to_str(model, p):
    """Represent value of *p*'th parameter of *model* as a string."""
    parameter = [param for param in model][p]
    # Represent P9 and P12 (scale parameters) in shorter form
    return parameter.value_str if p not in [8, 11] else ("%.3e" % parameter.value)

def update(fig):
    """Fit new pointing model and update plots."""
    # Perform early redraw to improve interactivity of clicks (which typically change state of target dots)
    # Target state: 0 = flagged, 1 = unflagged, 2 = highlighted
    #target_state = keep * ((target_index == fig.highlighted_target) + 1)
    target_state = 1
    # Specify colours of flagged, unflagged and highlighted dots, respectively, as RGBA tuples
    dot_colors = np.choose(target_state, np.atleast_3d(np.vstack([(1,1,1,1), (0,0,1,1), (1,0,0,1)]))).T
    for ax in fig.axes[:7]:
        ax.dots.set_facecolors(dot_colors)
    fig.canvas.draw()

    # Fit new pointing model and update results
    params, sigma_params = new_model.fit(az[keep], el[keep], measured_delta_az[keep], measured_delta_el[keep],
                                         std_delta_az[keep], std_delta_el[keep], enabled_params)
    new.update(new_model)

    # Update rest of figure
    fig.texts[3].set_text("$\chi^2$ = %.3e" % new.chi2)
    fig.texts[4].set_text("all sky rms = %.3f' (robust %.3f')" % (new.sky_rms, new.robust_sky_rms))
    new.metrics(target_index == fig.highlighted_target)
    fig.texts[5].set_text("target sky rms = %.3f' (robust %.3f')" % (new.sky_rms, new.robust_sky_rms))
    new.metrics(keep)
    #fig.texts[-1].set_text(unique_targets[fig.highlighted_target])
    # Update model parameter strings
    for p, param in enumerate(display_params):
        fig.texts[2*p + 6].set_text(param_to_str(new_model, param) if enabled_params[param] else '')
        # HACK to convert sigmas to arcminutes, but not for P9 and P12 (which are scale factors)
        # This functionality should really reside inside the PointingModel class
        std_param = rad2deg(sigma_params[param]) * 60. if param not in [8, 11] else sigma_params[param]
        std_param_str = ("%.2f'" % std_param) if param not in [8, 11] else ("%.0e" % std_param)
        fig.texts[2*p + 7].set_text(std_param_str if enabled_params[param] and opts.use_stats else '')
        # Turn parameter string bold if it changed significantly from old value
        if np.abs(params[param] - list(old_model.values())[param]) > 3.0 * sigma_params[param]:
            fig.texts[2*p + 6].set_weight('bold')
            fig.texts[2*p + 7].set_weight('bold')
        else:
            fig.texts[2*p + 6].set_weight('normal')
            fig.texts[2*p + 7].set_weight('normal')
    daz_az, del_az, daz_el, del_el, quiver, before, after = fig.axes[:7]
    # Update quiver plot
    quiver_scale = 0.1 * fig.quiver_scale_slider.val * np.pi / 6 / deg2rad(old.robust_sky_rms / 60.)
    quiver.quiv.set_segments(quiver_segments(new.residual_az, new.residual_el, quiver_scale))
    quiver.quiv.set_color(np.choose(keep, np.atleast_3d(np.vstack([(0.3,0.3,0.3,0.2), (0.3,0.3,0.3,1)]))).T)
    # Update residual plots
    daz_az.dots.set_offsets(np.c_[rad2deg(az), rad2deg(new.residual_xel) * 60.])
    del_az.dots.set_offsets(np.c_[rad2deg(az), rad2deg(new.residual_el) * 60.])
    daz_el.dots.set_offsets(np.c_[rad2deg(el), rad2deg(new.residual_xel) * 60.])
    del_el.dots.set_offsets(np.c_[rad2deg(el), rad2deg(new.residual_el) * 60.])
    after.dots.set_offsets(np.c_[np.arctan2(new.residual_el, new.residual_xel), new.abs_sky_error])
    resid_lim = 1.2 * max(new.abs_sky_error.max(), old.abs_sky_error.max())
    daz_az.set_ylim(-resid_lim, resid_lim)
    del_az.set_ylim(-resid_lim, resid_lim)
    daz_el.set_ylim(-resid_lim, resid_lim)
    del_el.set_ylim(-resid_lim, resid_lim)
    before.set_ylim(0, resid_lim)
    after.set_ylim(0, resid_lim)
    # Redraw the figure
    fig.canvas.draw()

def setup_param_button(p):
    """Set up individual parameter toggle button."""
    param = display_params[p]
    param_button = mpl.widgets.Button(fig.add_axes([0.09, 0.94 - (0.85 + p * 0.9) / len(display_params),
                                                   0.03, 0.85 / len(display_params)]), 'P%d' % (param + 1,))
    fig.text(0.19, 0.94 - (0.5 * 0.85 + p * 0.9) / len(display_params), '', ha='right', va='center')
    fig.text(0.24, 0.94 - (0.5 * 0.85 + p * 0.9) / len(display_params), '', ha='right', va='center')
    state = enabled_params[param]
    param_button.label.set_color(param_button_color[state])
    param_button.label.set_weight(param_button_weight[state])
    def toggle_param_callback(event):
        state = not enabled_params[param]
        enabled_params[param] = state
        param_button.label.set_color(param_button_color[state])
        param_button.label.set_weight(param_button_weight[state])
        update(fig)
#    param_button.on_clicked(toggle_param_callback)
    return param_button # This is to stop the gc from deleting the data

theta_formatter = PolarAxes.ThetaFormatter()
def angle_formatter(x, pos=None):
    return theta_formatter(wrap_angle(np.pi / 2.0 - x), pos)
def arcmin_formatter(x, pos=None):
    return "%g'" % x

def plot_data_and_tooltip(ax, xdata, ydata):
    """Plot data markers and add tooltip to axes (single place to configure them)."""
    ax.dots = ax.scatter(xdata, ydata, 30, 'b', edgecolors='0.3')
    ax.loupe = ax.plot(0, 0, 'o', ms=14, mfc='None', mew=3., visible=False)[0]
    ax.ann = ax.annotate('', xy=(0., 0.), xycoords='data', xytext=(32, 32), textcoords='offset points', size=14,
                         va='bottom', ha='center', bbox=dict(boxstyle='round4', fc='w'), visible=False, zorder=5,
                         arrowprops=dict(arrowstyle='-|>', shrinkB=10, connectionstyle='arc3,rad=-0.2',
                                         fc='w', zorder=4))

string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
now = time.strftime('%Y-%m-%d_%Hh%M') # current time stamp
parser = optparse.OptionParser(usage="%prog [options] <data  files > ",
                               description="This fits a pointing model to the given data CSV file"
                               " with the targets that are included in the the offset pointing csv file. "
                               " You will also be lucky enough to get a plot comparing the results "
                               " between the old and newly derived pointing models.")
parser.add_option('-c', '--compare', action='store_false', default=True,
                  help="Do not plot comparison between fit models.")
parser.add_option('-n', '--no-stats', dest='use_stats', action='store_false', default=True,
                  help="Ignore uncertainties of data points during fitting")
parser.add_option('-o', '--output', dest='outfilebase', default='pointing_model_%s' % (now,),
                  help="Base name of output files (*.csv for new pointing model and *_data.csv for residuals, "
                  "default is 'pointing_model_<time>')")
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
if len(args) < 1 or not args[0].endswith('.csv'):
    raise RuntimeError('Correct File not passed to program. File should be csv file')
filename = args[0]
data = None
for filename in args:
    if data is None:
        data = read_offsetfile(filename)
    else:
        data = np.r_[data,read_offsetfile(filename)]


# Choose Data 
target_list = np.unique(data['target'])
np.random.shuffle(target_list)
sample_number = np.floor(target_list.shape[0]*0.2).astype(int) # Choose 20% of the unique targets
offsetdata = target_list[0:sample_number]
keep = np.ones((len(data)),dtype=np.bool)
for key,target in enumerate(data['target']):
    keep[key] = target not in set(offsetdata)

i = 0
tmpstr = ""
linelength = 5
text.append("List of targets used:")
for  tar in list(set(data['target'])):
    if  i % linelength == linelength-1 :
        text.append(tmpstr)
        tmpstr = ""
    i = i + 1
    tmpstr +='%s, '%(tar)
text.append(tmpstr)

#####################################
# Load old pointing model, if given.
#####################################
old_model = None
if opts.pmfilename:
    try:
        old_model = katpoint.PointingModel(file(opts.pmfilename).readline())
        logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(old_model), opts.pmfilename))
    except IOError:
        logger.warning("Could not load old pointing model from '%s'" % (opts.pmfilename,))

# If the antenna has no model specified, a default null model will be used
antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])
if old_model is None:
    old_model = antenna.pointing_model
targets = data['target']
#keep = data['keep'].astype(np.bool) if 'keep' in data.dtype.fields else np.tile(True, len(targets))

##########################################
# Initialise new pointing model and set 
# default enabled parameters.
##########################################
new_model = katpoint.PointingModel()
num_params = len(new_model)
default_enabled = np.array([1, 3, 4, 5, 6, 7,8]) - 1
enabled_params = np.tile(False, num_params)
enabled_params[default_enabled] = True
enabled_params = enabled_params.tolist()

# For display purposes, throw out unused parameters P2 and P10
display_params = list(range(num_params))
display_params.pop(9)
display_params.pop(1)

# Fit new pointing model
az, el = angle_wrap(deg2rad(data['azimuth'])), deg2rad(data['elevation'])
measured_delta_az, measured_delta_el = deg2rad(data['delta_azimuth']), deg2rad(data['delta_elevation'])
# Uncertainties are optional
min_std = deg2rad(min_rms  / 60. / np.sqrt(2))
std_delta_az = np.clip(deg2rad(data['delta_azimuth_std']), min_std, np.inf) \
    if 'delta_azimuth_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(az))
std_delta_el = np.clip(deg2rad(data['delta_elevation_std']), min_std, np.inf) \
    if 'delta_elevation_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(el))

params, sigma_params = new_model.fit(az[keep], el[keep], measured_delta_az[keep], measured_delta_el[keep],
                                     std_delta_az[keep], std_delta_el[keep], enabled_params)

###############################################
# Perform calculations and assign text output.
###############################################
text.append("Blind Pointing metrics for fitted points. (N = %i  Fitting Data Points) "%(np.sum(keep)))

sky_rms,robust_sky_rms,chi2,text1 = metrics(new_model,az[keep],el[keep],measured_delta_az[keep], measured_delta_el[keep] ,std_delta_az[keep] ,std_delta_el[keep])
old = PointingResults(old_model)
new = PointingResults(new_model)
text += text1
text.append("")
text.append("Blind Pointing metrics for test points.  (N = %i Test Data Points) R.T.P.3"%(np.sum(~keep)))
i = 0
tmpstr = ""
linelength = 5
text.append("List of test targets used:")
for  tar in list(offsetdata):
    if  i % linelength == linelength-1 :
        text.append(tmpstr)
        tmpstr = ""
    i = i + 1
    tmpstr +='%s, '%(tar)
text.append(tmpstr)

sky_rms,robust_sky_rms,chi2,text1 = metrics(new_model,az[~keep],el[~keep],measured_delta_az[~keep], measured_delta_el[~keep] ,std_delta_az[~keep] ,std_delta_el[~keep])
text += text1
text.append("")
#print new_model.description

text.append("")
for line in text: print(line)

###################################
# Plot pointing model comparison.
##################################
nice_filename =  args[0].split('/')[-1].split('?')[0]+ '_pointing_stats'
pp = PdfPages(nice_filename+'.pdf')

if opts.compare:
    # List of unique targets in data set and target index for each data point
    unique_targets = np.unique(targets).tolist()
    target_index = np.array([unique_targets.index(t) for t in targets])
    # List of colors used to represent different targets in scatter plots
    scatter_colors = ('b', 'r', 'g', 'k', 'c', 'm', 'y')
    target_colors = np.tile(scatter_colors, 1 + len(unique_targets) // len(scatter_colors))[:len(unique_targets)]
    # Quantity loosely related to the declination of the source
    north = (np.pi / 2. - el) / (np.pi / 2.) * np.cos(az)
    pseudo_dec = -np.ones(len(unique_targets))
    for n, ind in enumerate(target_index):
        if north[n] > pseudo_dec[ind]:
            pseudo_dec[ind] = north[n]
    north_to_south = np.flipud(np.argsort(pseudo_dec))
    target_colors = target_colors[north_to_south][target_index]
    # Axis limit to be applied to all residual plots
    resid_lim = 1.2 * old.abs_sky_error.max()

    # make plot
    fig = plt.figure(1, figsize=(15, 10))
    fig.clear()
    # Store highlighted target index on figure object
    fig.highlighted_target = 0

    # Axes to contain detail residual plots - initialise plots with old residuals
    ax = fig.add_axes([0.27, 0.74, 0.2, 0.2])
    ax.axhline(0, color='k', zorder=0)
    plot_data_and_tooltip(ax, rad2deg(az), rad2deg(old.residual_xel) * 60.)
    ax.axis([-180., 180., -resid_lim, resid_lim])
    ax.set_xticks([])
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
    ax.set_ylabel('Cross-EL offset')
    ax.set_title('RESIDUALS')

    ax = fig.add_axes([0.27, 0.54, 0.2, 0.2])
    ax.axhline(0, color='k', zorder=0)
    plot_data_and_tooltip(ax, rad2deg(az), rad2deg(old.residual_el) * 60.)
    ax.axis([-180., 180., -resid_lim, resid_lim])
    ax.set_xlabel('Azimuth (deg)')
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
    ax.set_ylabel('EL offset')

    ax = fig.add_axes([0.27, 0.26, 0.2, 0.2])
    ax.axhline(0, color='k', zorder=0)
    plot_data_and_tooltip(ax, rad2deg(el), rad2deg(old.residual_xel) * 60.)
    ax.axis([0., 90., -resid_lim, resid_lim])
    ax.set_xticks([])
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
    ax.set_ylabel('Cross-EL offset')

    ax = fig.add_axes([0.27, 0.06, 0.2, 0.2])
    ax.axhline(0, color='k', zorder=0)
    plot_data_and_tooltip(ax, rad2deg(el), rad2deg(old.residual_el) * 60.)
    ax.axis([0., 90., -resid_lim, resid_lim])
    ax.set_xlabel('Elevation (deg)')
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
    ax.set_ylabel('EL offset')

    # Axes to contain quiver plot - plot static measurement locations in ARC projection as a start
    ax = fig.add_axes([0.5, 0.43, 0.5, 0.5], projection='polar')
    plot_data_and_tooltip(ax, np.pi/2. - az, np.pi/2. - el)
    segms = quiver_segments(old.residual_az, old.residual_el, 0.)
    ax.quiv = mpl.collections.LineCollection(segms, color='0.3')
    ax.add_collection(ax.quiv)
    ax.set_xticks(deg2rad(np.arange(0., 360., 90.)))
    ax.set_xticklabels(['E', 'N', 'W', 'S'])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(angle_formatter))
    ax.set_ylim(0., np.pi / 2.)
    ax.set_yticks(deg2rad(np.arange(0., 90., 10.)))
    ax.set_yticklabels([])

    # Axes to contain before/after residual plot
    ax = fig.add_axes([0.5, 0.135, 0.25, 0.25], projection='polar')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
    plot_data_and_tooltip(ax, np.arctan2(old.residual_el, old.residual_xel), old.abs_sky_error)
    ax.set_xticklabels([])
    ax.set_title('OLD')
    fig.text(0.625, 0.09, "$\chi^2$ = %.3e" % (old.chi2,), ha='center', va='baseline')
    fig.text(0.625, 0.06, "all sky rms = %.3f' (robust %.3f')" % (old.sky_rms, old.robust_sky_rms),
         ha='center', va='baseline')
    old.metrics(target_index == fig.highlighted_target)
    fig.text(0.625, 0.03, "target sky rms = %.3f' (robust %.3f')" % (old.sky_rms, old.robust_sky_rms),
         ha='center', va='baseline', fontdict=dict(color=(0.25,0,0,1)))
    old.metrics(keep)

    ax = fig.add_axes([0.75, 0.135, 0.25, 0.25], projection='polar')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
    plot_data_and_tooltip(ax, np.arctan2(old.residual_el, old.residual_xel), old.abs_sky_error)
    ax.set_xticklabels([])
    ax.set_title('NEW')
    fig.text(0.875, 0.09, "$\chi^2$ = %.1f" % (old.chi2,), ha='center', va='baseline')
    fig.text(0.875, 0.06, "all sky rms = %.3f' (robust %.3f')" % (old.sky_rms, old.robust_sky_rms),
         ha='center', va='baseline')
    old.metrics(target_index == fig.highlighted_target)
    fig.text(0.875, 0.03, "target sky rms = %.3f' (robust %.3f')" % (old.sky_rms, old.robust_sky_rms),
         ha='center', va='baseline', fontdict=dict(color=(0.25,0,0,1)))
    old.metrics(keep)

    # Create buttons to toggle parameter selection
    param_button_color = ['0.65', '0.0']
    param_button_weight = ['normal', 'bold']
    param_buttons = [setup_param_button(p) for p in range(len(display_params))]

    # Add old pointing model and labels
    list_o_names = 'Ant:%s , Datasets:'%(antenna.name) + ' ,'.join(np.unique(data['dataset']).tolist() )
    fig.text(0.405, 0.98,git_info(), horizontalalignment='right',fontsize=10)
    fig.text(0.905, 0.98,list_o_names, horizontalalignment='right',fontsize=10)
    fig.text(0.053, 0.95, 'OLD', ha='center', va='bottom', size='large')
    fig.text(0.105, 0.95, 'MODEL', ha='center', va='bottom', size='large')
    fig.text(0.16, 0.95, 'NEW', ha='center', va='bottom', size='large')
    fig.text(0.225, 0.95, 'STD', ha='center', va='bottom', size='large')
    for p, param in enumerate(display_params):
        param_str = param_to_str(old_model, param) if list(old_model.values())[param] else ''
        fig.text(0.085, 0.94 - (0.5 * 0.85 + p * 0.9) / len(display_params), param_str, ha='right', va='center')
    
    # Create quiver scale slider
    fig.quiver_scale_slider = mpl.widgets.Slider(fig.add_axes([0.9, 0.92, 0.07, 0.02]), 'Arrow scale', 0, 90, 10, '%.0f')
    update(fig)
    #plt.show()
    fig.savefig(pp,format='pdf')
    plt.close(fig)

if True:
    fig = plt.figure(None,figsize = (10,16))
    params = {'font.size': 12}
    plt.rcParams.update(params)
    ax = fig.add_subplot(111)
    anchored_text = AnchoredText('\n'.join(text), loc=2, frameon=False)
    ax.add_artist(anchored_text)
    ax.set_axis_off()
    #plt.figtext(0.08,0.7,'\n'.join(text),fontsize=12)
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    plt.subplots_adjust(top=0.99,bottom=0,right=0.975,left=0.01)
    fig.savefig(pp,format='pdf')
    pp.close()
    plt.close(fig)
    plt.close('all')
