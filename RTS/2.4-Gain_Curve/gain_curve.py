#!/usr/bin/python
# Read in the results produced by analyse_point_source_scans.py
# Perform gain curve calculations and produce plots for report.
# S Goedhart 31 Aug 2009

import os.path
import sys
import logging
import optparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy import optimize

import scape
import katpoint

parser = optparse.OptionParser(usage="%prog [opts] <directories or files>",
                               description="This fits gain curves to the results of analyse_point_source_scans.py")
parser.add_option("-o", "--output", dest="outfilebase", type="string", default='gain_curve',
                  help="Base name of output files (*.png for plots and *_results.txt for messages)")

(opts, args) = parser.parse_args()
if len(args) ==0:
    print 'Please specify a csv file with beam heights.'
    sys.exit(1)

filename = args[0]

I_plot_filename = opts.outfilebase + '_I.png'
VV_plot_filename = opts.outfilebase + '_VV.png'
HH_plot_filename = opts.outfilebase + '_HH.png'
results_filename = opts.outfilebase + '_results.txt'

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']

# Load data file in one shot as an array of strings
data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')

# Interpret first non-comment line as header
fields = data[0].tolist()

# Load antenna description string from first line of file and construct antenna object from it
antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])

# By default, all fields are assumed to contain floats
formats = np.tile('float32', len(fields))

# The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype

# Convert to heterogeneous record array
data = np.rec.fromarrays(data[1:].transpose(), dtype=zip(fields, formats))

# Obtain desired fields and convert to radians
az, el = angle_wrap(katpoint.deg2rad(data['azimuth'])), katpoint.deg2rad(data['elevation'])

# Make sure we only use data that had a successful noise diode calibration
# If the noise diode failed to fire the data unit stays as 'raw'
is_in_K = data['data_unit'] == 'K'

# Distinguish between different calibrator sources.
# 3C catalogue numbers used but can't start the variable names with a number
# Note that the current optimised gain curve is only using two calibrators
c48  = data['target'] == '3C 48'
c123 = data['target'] == '3C 123'
c161 = data['target'] == '3C 161'
c218 = data['target'] == 'Hyd A'
c274 = data['target'] == 'Vir A'
c286 = data['target'] == '3C 286'
c405 = data['target'] == 'Cyg A'
c353 = data['target'] == '3C 353'
PicA = data['target'] == 'Pic A'
HerA = data['target'] == 'Her A'
TauA = data['target'] == 'Tau A'
pks  = data['target'] == 'PKS 1934-63'

# Calculate gain for each observation. Gain = T_ant/Flux_density
gain_I = np.sqrt(data['beam_height_HH']*data['beam_height_VV']) / data['flux']
gain_HH = data['beam_height_HH'] / data['flux']
gain_VV = data['beam_height_VV'] / data['flux']

# Tsys can be estimated from the baseline height
Tsys_I = np.sqrt(data['baseline_height_HH']*data['baseline_height_VV'])
Tsys_HH = data['baseline_height_HH']
Tsys_VV = data['baseline_height_VV']

#Get the ambient atmospheric temperature.
T_atm = data['temperature']

# Dish geometrical area
A = np.pi * (antenna.diameter / 2.0) ** 2

#Aperture efficiency - before atmospheric fit.
e_I = gain_I*(2761/A)*100
e_HH = gain_HH*(2761/A)*100
e_VV = gain_VV*(2761/A)*100

#System Equivalent Flux Density
SEFD_I = Tsys_I/gain_I
SEFD_HH = Tsys_HH/gain_HH
SEFD_VV = Tsys_VV/gain_VV

# The data being used has been fitted using an automated routine.  RFI can mess up the fit.
# We know that the efficiency can not be greater than 100 % so use this to filter the data.
# Take a reasonable value for the Tsys cut-off.  This is also dependent on the baseline so will be 
# odd if the fit has gone wrong.
good_sources = c218  | c274        #We've currently settled on these.
good_I = (e_I < 100) & (e_I > 35) & (Tsys_I < 300) & ~(np.isnan(data['beam_height_I'] ))
good_HH = (e_HH < 100) & (e_HH > 35) & (Tsys_HH < 150) & ~(np.isnan(data['beam_height_HH'] ))
good_VV = (e_VV < 100) & (e_VV > 35) & (Tsys_VV < 150) & ~(np.isnan(data['beam_height_VV'] ))
good = is_in_K & good_I & good_HH & good_VV & good_sources

#Account for atmospheric elevation dependent effects:
#Atmospheric attenuation is given by
# G = G_0*exp(-tau*airmass)
airmass = 1/np.sin(np.radians(data['elevation'][good]))
#linearise the equation
log_g_I = np.log(gain_I[good])
log_g_HH = np.log(gain_HH[good])
log_g_VV = np.log(gain_VV[good])

atm_abs_fit_I = np.polyfit(airmass, log_g_I, 1)
atm_abs_fit_HH = np.polyfit(airmass, log_g_HH, 1)
atm_abs_fit_VV = np.polyfit(airmass, log_g_VV, 1)
g_0_I = np.exp(atm_abs_fit_I[1])
g_0_HH = np.exp(atm_abs_fit_HH[1])
g_0_VV = np.exp(atm_abs_fit_VV[1])
# By definition, tau is a positive value
tau_I = -atm_abs_fit_I[0]
tau_HH = -atm_abs_fit_HH[0]
tau_VV = -atm_abs_fit_VV[0]

#make plottable values of the fit
fit_elev = np.linspace(5,90,85,endpoint=False)
fit_I = g_0_I*np.exp(-tau_I/np.sin(np.radians(fit_elev)))
fit_HH = g_0_HH*np.exp(-tau_HH/np.sin(np.radians(fit_elev)))
fit_VV = g_0_VV*np.exp(-tau_VV/np.sin(np.radians(fit_elev)))

#Atmospheric emission is given by
# Tsys = T_rec + T_atm*(1-exp(-tau*airmass))
# linear eqn with x = 1-exp(-tau*airmass)
airmass = 1/np.sin(np.radians(data['elevation'][good]))
atm_em_fit_I = np.polyfit(1 - np.exp(-tau_I*airmass),Tsys_I[good],1)
atm_em_fit_HH = np.polyfit(1 - np.exp(-tau_HH*airmass),Tsys_HH[good],1)
atm_em_fit_VV = np.polyfit(1 - np.exp(-tau_VV*airmass),Tsys_VV[good],1)
T_rec_I = atm_em_fit_I[1]
T_rec_HH = atm_em_fit_HH[1]
T_rec_VV = atm_em_fit_VV[1]
T_atm_I = atm_em_fit_I[0]
T_atm_HH = atm_em_fit_HH[0]
T_atm_VV = atm_em_fit_VV[0]
#make plottable values of the fit
fit_Tsys_I = T_rec_I + T_atm_I*(1 - np.exp(-tau_I/np.sin(np.radians(fit_elev))))
fit_Tsys_HH = T_rec_HH + T_atm_HH*(1 - np.exp(-tau_HH/np.sin(np.radians(fit_elev))))
fit_Tsys_VV = T_rec_VV + T_atm_VV*(1 - np.exp(-tau_VV/np.sin(np.radians(fit_elev))))

#remove the effect of atmospheric attenuation from the aperture efficiency
corrected_e_I = (gain_I -  g_0_I*np.exp(-tau_I/np.sin(np.radians(data['elevation']))) + g_0_I)*(2761/A)*100
corrected_e_HH = (gain_HH -  g_0_HH*np.exp(-tau_HH/np.sin(np.radians(data['elevation']))) + g_0_HH)*(2761/A)*100
corrected_e_VV = (gain_VV -  g_0_VV*np.exp(-tau_VV/np.sin(np.radians(data['elevation']))) + g_0_VV)*(2761/A)*100

high_el = data['elevation'] > 20

#calculate Ae from G_0
#might be a more robust way of doing the calculation if the data is noisy.
#provided you can believe the atmospheric opacity, which I don't.
Ae_I = g_0_I*(2761/A)*100
Ae_HH = g_0_HH*(2761/A)*100
Ae_VV = g_0_VV*(2761/A)*100

#Report interesting results
#While one does not normally report gain curves per polarisation, they are included to get an idea of the 
#consistency of the results and might be an indication of gain instability in one of the input channels.
f = file(results_filename, 'w')
f.write('command line options used:%s %s\n '%(opts, args))
f.write('\n')
f.write('Results for Stokes I\n')
print('Results for Stokes I\n')
f.write('Tau\t G_0\t median_G\t std\t Ae\t T_rec\t T_atm\t median_e\t std\t T_sys\t std\t SEFD\t sigma\n')
print('Tau\t G_0\t median_G\t std\t Ae\t T_rec\t T_atm\t median_e\t std\t T_sys\t std\t SEFD\t sigma\n')
print('%.4f\t %.4f\t %.4f\t %.4f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n'%\
   (tau_I, g_0_I, np.median(gain_I[high_el&good]),np.std(gain_I[high_el&good]), Ae_I, T_rec_I, T_atm_I,\
    np.median(e_I[high_el&good]),np.std(e_I[high_el&good]), np.median(Tsys_I[high_el&good]),\
    np.std(Tsys_I[high_el&good]),np.median(SEFD_I[high_el&good]),np.std(SEFD_I[high_el&good])))
f.write('%.4f\t %.4f\t %.4f\t %.4f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n'%\
   (tau_I, g_0_I, np.median(gain_I[high_el&good]),np.std(gain_I[high_el&good]), Ae_I, T_rec_I, T_atm_I,\
    np.median(e_I[high_el&good]),np.std(e_I[high_el&good]), np.median(Tsys_I[high_el&good]),\
    np.std(Tsys_I[high_el&good]),np.median(SEFD_I[high_el&good]),np.std(SEFD_I[high_el&good])))
f.write('Results for HH pol\n')
print('Results for HH pol\n')
f.write('Tau\t G_0\t median_G\t std\t Ae\t T_rec\t T_atm\t median_e\t std\t T_sys\t std\t SEFD\t sigma\n')
print('Tau\t G_0\t median_G\t std\t Ae\t T_rec\t T_atm\t median_e\t std\t T_sys\t std\t SEFD\t sigma\n')
print('%.4f\t %.4f\t %.4f\t %.4f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n'%\
   (tau_HH, g_0_HH, np.median(gain_HH[high_el&good]),np.std(gain_HH[high_el&good]), Ae_HH, T_rec_HH, T_atm_HH,\
    np.median(e_HH[high_el&good]),np.std(e_HH[high_el&good]), np.median(Tsys_HH[high_el&good]),\
    np.std(Tsys_HH[high_el&good]),np.median(SEFD_HH[high_el&good]),np.std(SEFD_HH[high_el&good])))
f.write('%.4f\t %.4f\t %.4f\t %.4f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n'%\
   (tau_HH, g_0_HH, np.median(gain_HH[high_el&good]),np.std(gain_HH[high_el&good]), Ae_HH, T_rec_HH, T_atm_HH,\
    np.median(e_HH[high_el&good]),np.std(e_HH[high_el&good]), np.median(Tsys_HH[high_el&good]),\
    np.std(Tsys_HH[high_el&good]),np.median(SEFD_HH[high_el&good]),np.std(SEFD_HH[high_el&good])))
f.write('Results for VV pol\n')
print('Results for VV pol\n')
f.write('Tau\t G_0\t median_G\t std\t Ae\t T_rec\t T_atm\t median_e\t std\t T_sys\t std\t SEFD\t sigma\n')
print('Tau\t G_0\t median_G\t std\t Ae\t T_rec\t T_atm\t median_e\t std\t T_sys\t std\t SEFD\t sigma\n')
print('%.4f\t %.4f\t %.4f\t %.4f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n'%\
   (tau_VV, g_0_VV, np.median(gain_VV[high_el&good]),np.std(gain_VV[high_el&good]), Ae_VV, T_rec_VV, T_atm_VV,\
    np.median(e_VV[high_el&good]),np.std(e_VV[high_el&good]), np.median(Tsys_VV[high_el&good]),\
    np.std(Tsys_VV[high_el&good]),np.median(SEFD_VV[high_el&good]),np.std(SEFD_VV[high_el&good])))
f.write('%.4f\t %.4f\t %.4f\t %.4f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n'%\
   (tau_VV, g_0_VV, np.median(gain_VV[high_el&good]),np.std(gain_VV[high_el&good]), Ae_VV, T_rec_VV, T_atm_VV,\
    np.median(e_VV[high_el&good]),np.std(e_VV[high_el&good]), np.median(Tsys_VV[high_el&good]),\
    np.std(Tsys_VV[high_el&good]),np.median(SEFD_VV[high_el&good]),np.std(SEFD_VV[high_el&good])))
f.close()

#Plot results
fig = plt.figure()
fig.set_size_inches(12,8)
fig.subplots_adjust(hspace=0.1)


ax1 = plt.subplot(411)
plt.hold(True)
plt.plot(data['elevation'][c218&good], gain_I[c218&good] ,'co', label='3C218')
plt.plot(data['elevation'][c274&good], gain_I[c274&good] ,'go', label='3C274')
plt.plot(fit_elev, fit_I, 'k-')
plt.hold(False)
plt.ylabel('Gain (K/Jy)')
plt.title('Gain curve '+antenna.name)
legend = plt.legend(loc=7)
plt.setp(legend.get_texts(), fontsize='small')

ax2 = plt.subplot(412, sharex=ax1)
plt.hold(True)
plt.plot(data['elevation'][c218&good], e_I[c218&good] ,'co')
plt.plot(data['elevation'][c274&good], e_I[c274&good] ,'go')
plt.hold(False)
plt.ylabel('e  %')

ax3 = plt.subplot(413, sharex=ax1)
plt.hold(True)
plt.plot(data['elevation'][c218&good], Tsys_HH[c218&good] ,'bo', label='HH')
plt.plot(data['elevation'][c274&good], Tsys_HH[c274&good] ,'bo')
plt.plot(data['elevation'][c218&good], Tsys_VV[c218&good] ,'ro', label='VV')
plt.plot(data['elevation'][c274&good], Tsys_VV[c274&good] ,'ro')
plt.plot(fit_elev, fit_Tsys_HH, 'b-')
plt.plot(fit_elev, fit_Tsys_VV, 'r-')
legend = plt.legend(loc=7)
plt.setp(legend.get_texts(), fontsize='small')
plt.hold(False)
plt.ylabel('Tsys (K)')

ax4 = plt.subplot(414, sharex=ax1)
plt.hold(True)
plt.plot(data['elevation'][c218&good], SEFD_I[c218&good] ,'co')
plt.plot(data['elevation'][c274&good], SEFD_I[c274&good] ,'go')
plt.hold(False)
plt.ylabel('SEFD (Jy)')
plt.xlabel('Elevation (deg)')
xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()
plt.setp(xticklabels, visible=False)
fig.savefig(I_plot_filename, dpi=200, bbox_inches='tight')
fig.show()


