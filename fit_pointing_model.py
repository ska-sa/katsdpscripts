#! /usr/bin/python
# Example script that fits pointing model to point source scan data product.
#
# First run the analyse_point_source_scans.py script to generate the data file
# that serves as input to this script.
#
# Ludwig Schwardt
# 26 August 2009
#

import sys
import optparse
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.projections import PolarAxes

import katpoint
from katpoint import rad2deg, deg2rad

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
# Create a date/time string for current time
now = time.strftime('%Y-%m-%d_%Hh%M')

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file>",
                               description="This fits a pointing model to the given data file. \
                                            It runs interactively by default, which allows the user \
                                            to select which parameters to fit and to inspect results.")
parser.set_defaults(pmfilename='pointing_model.csv', outfilename='pointing_model_%s.csv' % (now,))
parser.add_option("-b", "--batch", dest="batch", action="store_true",
                  help="True if processing is to be done in batch mode without user interaction")
parser.add_option("-p", "--pointing_model", dest="pmfilename", type="string",
                  help="Name of optional file containing old pointing model")
parser.add_option("-o", "--output", dest="outfilename", type="string",
                  help="Name of output file containing new pointing model")

(options, args) = parser.parse_args()
if len(args) < 1:
    print 'Please specify the name of data file to process'
    sys.exit(1)
filename = args[0]

# Set up logging: logging everything (DEBUG & above)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.root
logger.setLevel(logging.DEBUG)

# Load old pointing model
try:
    old_model = file(options.pmfilename).readline().strip()
    logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(old_model.split(',')), options.pmfilename))
    old_model = katpoint.PointingModel(old_model, strict=False)
except IOError:
    old_model = None

# Load data file in one shot as an array of strings
data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
# Interpret first non-comment line as header
fields = data[0].tolist()
# By default, all fields are assumed to contain floats
formats = np.tile('float32', len(fields))
# The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
# Convert to heterogeneous record array
data = np.rec.fromarrays(data[1:].transpose(), dtype=zip(fields, formats))
# Load antenna description string from first line of file and construct antenna object from it
antenna = katpoint.Antenna(file(filename).readline().strip().split('=')[1])
if old_model is None:
    old_model = antenna.pointing_model

# Obtain desired fields and convert to radians
az, el = angle_wrap(deg2rad(data['azimuth'])), deg2rad(data['elevation'])
measured_delta_az = deg2rad(data['delta_azimuth'])
measured_delta_el = deg2rad(data['delta_elevation'])
targets = data['target']
# List of unique targets in data set and target index for each data point
unique_targets = np.unique(targets).tolist()
target_indices = np.array([unique_targets.index(t) for t in targets])
current_target = 0
selected_target = target_indices == current_target
fwhm_beamwidth = 1.178 * katpoint.lightspeed / (data['frequency'][0] * 1e6) / antenna.diameter
beam_radius_arcmin = rad2deg(fwhm_beamwidth) * 60. / 2

# Previous residual error and sky RMS
old_residual_az, old_residual_el = measured_delta_az, measured_delta_el
if old_model:
    old_model_delta_az, old_model_delta_el = old_model.offset(az, el)
    old_residual_az = old_residual_az - old_model_delta_az
    old_residual_el = old_residual_el - old_model_delta_el
old_residual_xel = old_residual_az * np.cos(el)
old_res_x, old_res_y = rad2deg(old_residual_xel) * 60., rad2deg(old_residual_el) * 60.
old_abs_sky_error = np.sqrt(old_res_x * old_res_x + old_res_y * old_res_y)
old_sky_rms = np.sqrt(np.mean(old_abs_sky_error ** 2))
old_sky_rms = np.median(old_abs_sky_error) * np.sqrt(2. / np.log(4.))

# Data structure for quiver lines, consisting of arc coords and NaNs for the gaps between lines
quiver_scale = 10. * 60. / np.median(old_abs_sky_error)
line_sweep = np.linspace(0., 1., 21)
line_sweep[-1] = np.nan
quiver_theta = np.pi / 2. - az[:, np.newaxis] - quiver_scale * np.outer(old_residual_az, line_sweep)
quiver_r = np.pi / 2. - el[:, np.newaxis] - quiver_scale * np.outer(old_residual_el, line_sweep)

# Initialise new pointing model and set default enabled parameters
new_model = katpoint.PointingModel()
num_params = new_model.num_params
if old_model:
    default_enabled = old_model.params.nonzero()[0]
else:
    default_enabled = np.array([1, 3, 4, 5, 6, 7]) - 1
enabled_params = np.tile(False, num_params)
enabled_params[default_enabled] = True
enabled_params = enabled_params.tolist()

# Provide access to new residuals
residual_az, residual_el = None, None

def update(fig=None):
    """Fit new pointing model and update plots."""
    global residual_az, residual_el
    # Fit new pointing model
    params, sigma_params = new_model.fit(az, el, measured_delta_az, measured_delta_el, enabled_params=enabled_params)
    # Determine new residuals and sky RMS
    model_delta_az, model_delta_el = new_model.offset(az, el)
    residual_az, residual_el = measured_delta_az - model_delta_az, measured_delta_el - model_delta_el
    residual_xel = residual_az * np.cos(el)
    res_x, res_y = rad2deg(residual_xel) * 60., rad2deg(residual_el) * 60.
    abs_sky_error = np.sqrt(res_x * res_x + res_y * res_y)
    sky_rms = np.sqrt(np.mean(abs_sky_error ** 2))
    sky_rms = np.median(abs_sky_error) * np.sqrt(2. / np.log(4.))
    # Select new target data
    selected_target = target_indices == current_target
    target_sky_rms = np.median(abs_sky_error[selected_target]) * np.sqrt(2. / np.log(4.))
    target_old_sky_rms = np.median(old_abs_sky_error[selected_target]) * np.sqrt(2. / np.log(4.))

    # Update figure if not running in batch mode
    if fig:
        fig.texts[1].set_text("target sky rms = %.3f'" % target_old_sky_rms)
        fig.texts[2].set_text("all sky rms = %.3f'" % sky_rms)
        fig.texts[3].set_text("target sky rms = %.3f'" % target_sky_rms)
        fig.texts[-1].set_text(unique_targets[current_target])
        # Update model parameter strings
        for p in xrange(num_params):
            fig.texts[p + 4].set_text(new_model.param_str(p + 1) if enabled_params[p] else '')
        daz_az, del_az, daz_el, del_el, quiver, before, after = fig.axes[:7]
        # Update quiver plot
        quiver_theta[:] = np.pi / 2. - az[:, np.newaxis] - quiver_scale * np.outer(residual_az, line_sweep)
        quiver_r[:] = np.pi / 2. - el[:, np.newaxis] - quiver_scale * np.outer(residual_el, line_sweep)
        quiver.lines[1].set_data(quiver_theta.ravel(), quiver_r.ravel())
        quiver.lines[2].set_data(np.pi / 2. - az[selected_target], np.pi / 2. - el[selected_target])
        quiver.lines[3].set_data(quiver_theta[selected_target, :].ravel(), quiver_r[selected_target, :].ravel())
        # Update residual plots
        daz_az.lines[1].set_ydata(residual_xel)
        daz_az.lines[2].set_data(rad2deg(az[selected_target]), residual_xel[selected_target])
        del_az.lines[1].set_ydata(residual_el)
        del_az.lines[2].set_data(rad2deg(az[selected_target]), residual_el[selected_target])
        daz_el.lines[1].set_ydata(residual_xel)
        daz_el.lines[2].set_data(rad2deg(el[selected_target]), residual_xel[selected_target])
        del_el.lines[1].set_ydata(residual_el)
        del_el.lines[2].set_data(rad2deg(el[selected_target]), residual_el[selected_target])
        before.lines[2].set_data(np.arctan2(old_res_y, old_res_x)[selected_target], old_abs_sky_error[selected_target])
        after.lines[1].set_data(np.arctan2(res_y, res_x), abs_sky_error)
        after.lines[2].set_data(np.arctan2(res_y, res_x)[selected_target], abs_sky_error[selected_target])
        # Redraw all plots
        plt.draw()

### BATCH MODE ###

# This will fit the pointing model and quit
if options.batch:
    update()
    sys.exit(0)

### INTERACTIVE MODE ###

theta_formatter = PolarAxes.ThetaFormatter()
def angle_formatter(x, pos=None):
    return theta_formatter(angle_wrap(np.pi / 2.0 - x), pos)
def arcmin_formatter(x, pos=None):
    return "%g'" % x

# Set up figure with buttons
plt.ion()
fig = plt.figure(1)
plt.clf()
# Axes to contain detail residual plots - initialise plots with old residuals
plt.axes([0.15, 0.74, 0.3, 0.2])
plt.axhline(0, color='k')
plt.plot(rad2deg(az), old_residual_xel, 'ob')
plt.plot(rad2deg(az[selected_target]), old_residual_xel[selected_target], 'or')
ymax = 1.1 * np.abs(plt.ylim()).max()
plt.axis([-180., 180., -ymax, ymax])
plt.xticks([])
plt.yticks([])
plt.ylabel('Cross-EL offset')
plt.title('RESIDUALS')

plt.axes([0.15, 0.54, 0.3, 0.2])
plt.axhline(0, color='k')
plt.plot(rad2deg(az), old_residual_el, 'ob')
plt.plot(rad2deg(az[selected_target]), old_residual_el[selected_target], 'or')
ymax = 1.1 * np.abs(plt.ylim()).max()
plt.axis([-180., 180., -ymax, ymax])
plt.xlabel('Azimuth (deg)')
plt.yticks([])
plt.ylabel('EL offset')

plt.axes([0.15, 0.26, 0.3, 0.2])
plt.axhline(0, color='k')
plt.plot(rad2deg(el), old_residual_xel, 'ob')
plt.plot(rad2deg(el[selected_target]), old_residual_xel[selected_target], 'or')
ymax = 1.1 * np.abs(plt.ylim()).max()
plt.axis([0., 90., -ymax, ymax])
plt.xticks([])
plt.yticks([])
plt.ylabel('Cross-EL offset')

plt.axes([0.15, 0.06, 0.3, 0.2])
plt.axhline(0, color='k')
plt.plot(rad2deg(el), old_residual_el, 'ob')
plt.plot(rad2deg(el[selected_target]), old_residual_el[selected_target], 'or')
ymax = 1.1 * np.abs(plt.ylim()).max()
plt.axis([0., 90., -ymax, ymax])
plt.xlabel('Elevation (deg)')
plt.yticks([])
plt.ylabel('EL offset')

# Axes to contain quiver plot - plot static measurement locations in ARC projection as a start
# Resolution has to be 1 to prevent lines crossing the theta=0 border from wrapping around the wrong way
ax1 = plt.axes([0.5, 0.4, 0.5, 0.5], polar=True, resolution=1)
ax1.plot(np.pi /2 - az, np.pi/2. - el, 'ob')
ax1.plot(quiver_theta.ravel(), quiver_r.ravel(), 'k')
ax1.plot(np.pi /2 - az[selected_target], np.pi/2. - el[selected_target], 'or')
ax1.plot(quiver_theta[selected_target, :].ravel(), quiver_r[selected_target, :].ravel(), 'r')
# Manually add elevation grid lines, as the resolution=1 kwarg seems to mess up normal polar grid lines
for ytick in ax1.get_yticks():
    ax1.plot(np.linspace(0, 2 * np.pi, 50), [ytick] * 50, linestyle=':', color='k', lw=2)
ax1.set_xticks(deg2rad(np.arange(0., 360., 90.)))
ax1.set_xticklabels(['E', 'N', 'W', 'S'])
ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(angle_formatter))
ax1.set_ylim(0., np.pi / 2.)
ax1.set_yticks(deg2rad(np.arange(0., 90., 10.)))

# Axes to contain before/after residual plot
ax2 = plt.axes([0.5, 0.1, 0.25, 0.25], polar=True)
ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
ax2.plot([0, 2 * np.pi], [beam_radius_arcmin, beam_radius_arcmin], '--k')
ax2.plot(np.arctan2(old_res_y, old_res_x), old_abs_sky_error, 'ob')
ax2.plot(np.arctan2(old_res_y, old_res_x)[selected_target], old_abs_sky_error[selected_target], 'or')
ax2.set_ylim(0, 2 * beam_radius_arcmin)
ax2.set_xticklabels([])
plt.title('BEFORE')
plt.figtext(0.625, 0.07, "all sky rms = %.3f'" % (old_sky_rms,), ha='center', va='bottom')
plt.figtext(0.625, 0.04, "target sky rms = %.3f'" % (old_sky_rms,), ha='center', va='bottom')

ax3 = plt.axes([0.75, 0.1, 0.25, 0.25], polar=True)
ax3.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(arcmin_formatter))
ax3.plot([0, 2 * np.pi], [beam_radius_arcmin, beam_radius_arcmin], '--k')
ax3.plot(np.arctan2(old_res_y, old_res_x), old_abs_sky_error, 'ob')
ax3.plot(np.arctan2(old_res_y, old_res_x)[selected_target], old_abs_sky_error[selected_target], 'or')
ax3.set_ylim(0, 2 * beam_radius_arcmin)
ax3.set_xticklabels([])
plt.title('AFTER')
plt.figtext(0.875, 0.07, "all sky rms = %.3f'" % (old_sky_rms,), ha='center', va='bottom')
plt.figtext(0.875, 0.04, "target sky rms = %.3f'" % (old_sky_rms,), ha='center', va='bottom')

# Create save button
save_button = mpl.widgets.Button(plt.axes([0.495, 0.8, 0.05, 0.04]), 'SAVE',
                                 color=(0.85, 0, 0), hovercolor=(0.95, 0, 0))
def save_callback(event):
    outfile = file(options.outfilename, 'w')
    outfile.write(new_model.description)
    outfile.close()
    logger.debug("Saved %d-parameter pointing model to '%s'" % (len(new_model.params), options.outfilename))
    save_button.color = '0.85'
    save_button.hovercolor = '0.95'
save_button.on_clicked(save_callback)

# Create buttons to toggle parameter selection
param_button_color = ['0.65', '0.0']
param_button_weight = ['normal', 'bold']
def setup_param_button(p):
    """Set up individual parameter toggle button."""
    param_button = mpl.widgets.Button(plt.axes([0.02, 0.94 - (0.85 + p * 0.9) / num_params,
                                                0.03, 0.85 / num_params]), 'P%d' % (p + 1,))
    plt.figtext(0.06, 0.94 - (0.5 * 0.85 + p * 0.9) / num_params, '', va='center')
    state = enabled_params[p]
    param_button.label.set_color(param_button_color[state])
    param_button.label.set_weight(param_button_weight[state])
    def toggle_param_callback(event):
        state = not enabled_params[p]
        enabled_params[p] = state
        param_button.label.set_color(param_button_color[state])
        param_button.label.set_weight(param_button_weight[state])
        save_button.color = (0.85, 0, 0)
        save_button.hovercolor = (0.95, 0, 0)
        update(fig)
    param_button.on_clicked(toggle_param_callback)
param_buttons = [setup_param_button(p) for p in xrange(num_params)]
plt.figtext(0.05, 0.95, 'MODEL', ha='center', va='bottom', size='large')

# Create target selector buttons and related text (title + target string)
plt.figtext(0.55, 0.95, 'TARGET', ha='center', va='bottom', size='large')
prev_target_button = mpl.widgets.Button(plt.axes([0.495, 0.9, 0.05, 0.04]), 'PREV')
def prev_target_callback(event):
    global current_target
    current_target = current_target - 1 if current_target > 0 else len(unique_targets) - 1
    update(fig)
prev_target_button.on_clicked(prev_target_callback)
next_target_button = mpl.widgets.Button(plt.axes([0.555, 0.9, 0.05, 0.04]), 'NEXT')
def next_target_callback(event):
    global current_target
    current_target = current_target + 1 if current_target < len(unique_targets) - 1 else 0
    update(fig)
next_target_button.on_clicked(next_target_callback)
plt.figtext(0.55, 0.89, unique_targets[current_target], ha='center', va='top')

# Create quiver scale slider
quiver_scale_slider = mpl.widgets.Slider(plt.axes([0.85, 0.91, 0.1, 0.02]), 'Arrow scale', 0., 90., 10., '%.0f')
def quiver_scale_callback(event):
    global quiver_scale, quiver_theta, quiver_r
    quiver_scale = quiver_scale_slider.val * 60. / np.median(old_abs_sky_error)
    quiver_theta[:] = np.pi / 2. - az[:, np.newaxis] - quiver_scale * np.outer(residual_az, line_sweep)
    quiver_r[:] = np.pi / 2. - el[:, np.newaxis] - quiver_scale * np.outer(residual_el, line_sweep)
    fig.axes[4].lines[1].set_data(quiver_theta.ravel(), quiver_r.ravel())
    selected_target = target_indices == current_target
    fig.axes[4].lines[3].set_data(quiver_theta[selected_target, :].ravel(), quiver_r[selected_target, :].ravel())
    plt.draw()
quiver_scale_slider.on_changed(quiver_scale_callback)

# Start off the processing and hand over control to the main GUI loop
update(fig)
# Display plots - this should be called ONLY ONCE, at the VERY END of the script
# The script stops here until you close the plots...
plt.show()
